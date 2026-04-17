"""Adaptive method selection file. Evaluates all detectors, selects best one"""

import os, sys, json, argparse
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
import numpy as np
from pathlib import Path
from collections import Counter
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification

#Args
parser = argparse.ArgumentParser()
parser.add_argument("--artifact-root", required=True)
parser.add_argument("--output-dir", required=True)
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument("--max-length", type=int, default=512)
parser.add_argument("--target-fpr", type=float, default=0.05)
args = parser.parse_args()

ARTIFACT_ROOT = Path(args.artifact_root)
OUTPUT_DIR = Path(args.output_dir)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
BATCH_SIZE = args.batch_size
MAX_LENGTH = args.max_length
TARGET_FPR = args.target_fpr
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16

#Load prompts in
ds_dir = ARTIFACT_ROOT / "datasets" / "generative_detection" / "publication_detection"
gen_dir = ARTIFACT_ROOT / "generations" / "publication_detection"
models_dir = ARTIFACT_ROOT / "models" / "detectors"

with open(ds_dir / "prompts.jsonl") as f:
    all_prompts = [json.loads(line) for line in f]

all_prompts_by_id = {p["prompt_id"]: p for p in all_prompts}
test_ids = set(p["prompt_id"] for p in all_prompts if p["split"] == "test")
train_ids = set(p["prompt_id"] for p in all_prompts if p["split"] == "train")
print(f"Prompts: {len(all_prompts)} total, {len(train_ids)} train, {len(test_ids)} test")

#Load all method generations
methods = sorted([p.stem for p in gen_dir.glob("*.jsonl")
                  if not p.stem.startswith(".") and p.stem != "ADAPTIVE_SELECTION"])
print(f"Methods ({len(methods)}): {methods}")

prompt_rows = {}
for method in methods:
    with open(gen_dir / f"{method}.jsonl") as f:
        for line in f:
            row = json.loads(line)
            pid = row["prompt_id"]
            if pid not in prompt_rows:
                prompt_rows[pid] = {}
            prompt_rows[pid][method] = row

#Derive model pair
with open(gen_dir / "manifest.json") as f:
    gen_manifest = json.load(f)
gen_model = gen_manifest["publication_model_id"]
main_model = gen_model if "-Instruct" not in gen_model else gen_model.replace("-Instruct", "")
aux_model = gen_model + "-Instruct" if "-Instruct" not in gen_model else gen_model
print(f"Binoculars pair: {main_model} (q) + {aux_model} (r)")

#Load Binoculars models
token = os.environ.get("HF_TOKEN")
print(f"\nLoading {main_model}...")
tokenizer = AutoTokenizer.from_pretrained(main_model, token=token)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model_q = AutoModelForCausalLM.from_pretrained(main_model, torch_dtype=DTYPE, token=token).to(DEVICE)
model_q.eval()
print(f"Loading {aux_model}...")
model_r = AutoModelForCausalLM.from_pretrained(aux_model, torch_dtype=DTYPE, token=token).to(DEVICE)
model_r.eval()


def score_both_batch(texts, desc="scoring"):
    """
    Function that can execute both Binoculars and FastDetectGPT scores in a single forward pass
    """
    bino_scores, fdgpt_scores = [], []
    for start in tqdm(range(0, len(texts), BATCH_SIZE), desc=desc, unit="batch", leave=False):
        batch = texts[start:start + BATCH_SIZE]
        encoded = tokenizer(batch, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
        encoded = {k: v.to(DEVICE) for k, v in encoded.items()}

        with torch.no_grad():
            logits_q = model_q(**encoded).logits[:, :-1, :]
            logits_r = model_r(**encoded).logits[:, :-1, :]

        lab = encoded["input_ids"][:, 1:]
        mask = encoded["attention_mask"][:, 1:].float()
        lp = torch.log_softmax(logits_q, dim=-1)
        pr = torch.softmax(logits_r, dim=-1)
        sl = mask.sum(dim=-1).clamp(min=1.0)

        # Binoculars: Negative Log Likelihood and Cross Entropy
        nll = -torch.gather(lp, dim=-1, index=lab.unsqueeze(-1)).squeeze(-1)
        ce = -(pr * lp).sum(dim=-1)
        bino = ((nll * mask).sum(-1) / sl) / ((ce * mask).sum(-1) / sl).clamp(min=1e-8)
        bino_scores.extend(bino.cpu().tolist())


        # FastDetectGPT: (actual - expected) / sqrt(var)

        actual = torch.gather(lp, dim=-1, index=lab.unsqueeze(-1)).squeeze(-1)
        expected = (pr * lp).sum(dim=-1)
        var = ((pr * lp**2).sum(dim=-1) - expected**2).clamp(min=1e-8)
        num = ((actual - expected) * mask).sum(dim=-1)
        fdg = num / (var * mask).sum(dim=-1).clamp(min=1e-8).sqrt()
        fdgpt_scores.extend(fdg.cpu().tolist())

    return bino_scores, fdgpt_scores


def binoculars_score_batch(texts, desc="Binoculars"):
    '''
    Calculates scores for Binoculars
    '''
    return score_both_batch(texts, desc)[0]


def fastdetectgpt_score_batch(texts, desc="FastDetectGPT"):
    '''
    Calculates scores for FastDetectGPT
    '''
    return score_both_batch(texts, desc)[1]


#Scoring all prompts with Binoculars + FastDetectGPT
print("\n=== Scoring all texts with Binoculars + FastDetectGPT for selection ===")
prompt_bino = {}   
prompt_fdgpt = {}
for method in methods:
    texts, pids = [], []

    for pid in sorted(prompt_rows.keys()):
        if method in prompt_rows[pid]:
            texts.append(prompt_rows[pid][method]["text"])
            pids.append(pid)

    if not texts:
        continue

    b_sc, f_sc = score_both_batch(texts, desc=f"  {method}")
    for pid, b, f in zip(pids, b_sc, f_sc):
        prompt_bino.setdefault(pid, {})[method] = b
        prompt_fdgpt.setdefault(pid, {})[method] = f
    print(f"  {method}: bino={np.mean(b_sc):.4f}, fdgpt={np.mean(f_sc):.4f}")


def build_selection(prompt_scores: dict, direction: str) -> tuple[dict, Counter]:
    """
    Picks best model based off of detector - Binoculars wants highest for 'most human', and viceversa for FastD
    
    """
    cnt = Counter()
    sel = {}
    picker = max if direction == "max" else min
    for pid in sorted(prompt_rows.keys()):
        sc = prompt_scores.get(pid, {})
        if not sc:
            continue
        best = picker(sc, key=sc.get)
        cnt[best] += 1
        sel[pid] = (best, prompt_rows[pid][best]["text"], sc[best])
    return sel, cnt

#Pick highest for binoculars, lowest for FastDetect
selections_bino, counter_bino = build_selection(prompt_bino, "max")
selections_fdgpt, counter_fdgpt = build_selection(prompt_fdgpt, "min")

print("\n=== Binoculars-oracle distribution ===")
for m, c in counter_bino.most_common():
    print(f"  {m:<25} {c:>4} ({100*c/len(selections_bino):.1f}%)")
print("\n=== FastDetectGPT-oracle distribution ===")
for m, c in counter_fdgpt.most_common():
    print(f"  {m:<25} {c:>4} ({100*c/len(selections_fdgpt):.1f}%)")

#Prepare test texts (same prompt ids for both oracles)
test_pids = sorted(pid for pid in selections_bino if pid in test_ids)
human_texts = [all_prompts_by_id[pid]["reference_text"] for pid in test_pids]
selected_texts_bino = [selections_bino[pid][1] for pid in test_pids]
selected_texts_fdgpt = [selections_fdgpt[pid][1] for pid in test_pids]
n = len(test_pids)
labels = [0] * n + [1] * n
print(f"\nTest set: {n} prompts (per oracle)")


def compute_metrics(human_scores, ai_scores, score_multiplier=1.0):
    """Compute AUROC, accuracy, F1, FPR, TPR at target FPR."""
    h = [score_multiplier * s for s in human_scores]
    a = [score_multiplier * s for s in ai_scores]
    all_s = h + a
    auroc = roc_auc_score(labels, all_s)

    # Threshold at target FPR
    sorted_h = sorted(h)
    idx = min(max(int(len(sorted_h) * (1.0 - TARGET_FPR)), 0), len(sorted_h) - 1)
    threshold = sorted_h[idx]
    preds = [1 if s >= threshold else 0 for s in all_s]
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, zero_division=0)
    fp = sum(1 for i in range(n) if preds[i] == 1)
    tp = sum(1 for i in range(n, 2*n) if preds[i] == 1)
    fpr = fp / max(1, n)
    tpr = tp / max(1, n)
    mean_ai = float(np.mean(a))
    
    return {
        "auroc": auroc, "accuracy": acc, "f1": f1,
        "mean_ai_prob": mean_ai, "threshold": threshold,
        "fpr": fpr, "tpr": tpr,
    }


#Evaluate human and both selected corpora with for both unsupervised detectors
print("\n=== Evaluating unsupervised detectors on both corpora ===")
human_bino, human_fdgpt = score_both_batch(human_texts, desc="human")
bino_b, fdgpt_b = score_both_batch(selected_texts_bino, desc="sel_bino")
bino_f, fdgpt_f = score_both_batch(selected_texts_fdgpt, desc="sel_fdgpt")

#Binoculars-oracle corpus
metrics_bino_bino = compute_metrics(human_bino, bino_b, score_multiplier=-1.0)
metrics_bino_bino["metadata"] = {"score_multiplier": -1.0, "target_fpr": TARGET_FPR}
metrics_bino_fdgpt = compute_metrics(human_fdgpt, fdgpt_b, score_multiplier=1.0)
metrics_bino_fdgpt["metadata"] = {"score_multiplier": 1.0, "target_fpr": TARGET_FPR}
print(f"ADAPTIVE_BINOCULARS  | Binoculars AUROC: {metrics_bino_bino['auroc']:.4f}  "
      f"FastDetectGPT AUROC: {metrics_bino_fdgpt['auroc']:.4f}")

#FastDetectGPT-oracle corpus
metrics_fdgpt_bino = compute_metrics(human_bino, bino_f, score_multiplier=-1.0)
metrics_fdgpt_bino["metadata"] = {"score_multiplier": -1.0, "target_fpr": TARGET_FPR}
metrics_fdgpt_fdgpt = compute_metrics(human_fdgpt, fdgpt_f, score_multiplier=1.0)
metrics_fdgpt_fdgpt["metadata"] = {"score_multiplier": 1.0, "target_fpr": TARGET_FPR}
print(f"ADAPTIVE_FASTDETECT  | Binoculars AUROC: {metrics_fdgpt_bino['auroc']:.4f}  "
      f"FastDetectGPT AUROC: {metrics_fdgpt_fdgpt['auroc']:.4f}")

#Free LM models, load supervised
print("\nFreeing LM models")
del model_q, model_r
import gc; gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

#Evaluate with supervised detectors (score both corpora in one load per ckpt)
supervised_results = []
#Layout of combined tensor: [human (n), bino-selected (n), fdgpt-selected (n)]
combined_texts = human_texts + selected_texts_bino + selected_texts_fdgpt

def _metrics_from_scores(human_sc, ai_sc, threshold):
    '''
    Obtain scoring metrics for different methods
    '''
    all_s = human_sc + ai_sc
    auroc = roc_auc_score(labels, all_s)
    preds = [1 if s >= threshold else 0 for s in all_s]
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, zero_division=0)
    fp = sum(1 for i in range(n) if preds[i] == 1)
    tp = sum(1 for i in range(n, 2*n) if preds[i] == 1)
    return {
        "auroc": auroc, "accuracy": acc, "f1": f1,
        "mean_ai_prob": float(np.mean(ai_sc)),
        "threshold": threshold,
        "fpr": fp / max(1, n), "tpr": tp / max(1, n),
    }

for arch_dir in sorted(models_dir.iterdir()):
    if not arch_dir.is_dir() or not arch_dir.name.startswith("tuned-"):
        continue
    det_name = arch_dir.name.replace("tuned-", "")
    source_dirs = sorted([p for p in arch_dir.iterdir() if p.is_dir()])

    for ckpt in source_dirs:
        source = ckpt.name
        print(f"\n=== {det_name} ({source}) ===")
        try:
            sup_tokenizer = AutoTokenizer.from_pretrained(str(ckpt), fix_mistral_regex=True)
        except TypeError:
            sup_tokenizer = AutoTokenizer.from_pretrained(str(ckpt))
        sup_model = AutoModelForSequenceClassification.from_pretrained(str(ckpt)).to(DEVICE)
        sup_model.eval()

        metrics_file = ckpt / "training_metrics.json"
        threshold = 0.5
        if metrics_file.exists():
            with open(metrics_file) as f:
                tm = json.load(f)
            threshold = tm.get("decision_threshold", 0.5)

        scores = []
        for start in tqdm(range(0, len(combined_texts), BATCH_SIZE), desc=f"{det_name}", unit="batch", leave=False):
            batch = combined_texts[start:start + BATCH_SIZE]
            encoded = sup_tokenizer(batch, truncation=True, padding="max_length",
                                    max_length=MAX_LENGTH, return_tensors="pt")
            encoded = {k: v.to(DEVICE) for k, v in encoded.items()}
            with torch.no_grad():
                logits = sup_model(**encoded).logits
            probs = torch.softmax(logits, dim=-1)[:, 1].cpu().tolist()
            scores.extend(probs)

        human_sc = scores[:n]
        bino_sel_sc = scores[n:2*n]
        fdgpt_sel_sc = scores[2*n:3*n]

        for test_method, ai_sc in [
            ("ADAPTIVE_BINOCULARS", bino_sel_sc),
            ("ADAPTIVE_FASTDETECT", fdgpt_sel_sc),
        ]:
            m = _metrics_from_scores(human_sc, ai_sc, threshold)
            supervised_results.append({
                "train_method": source,
                "test_method": test_method,
                "detector_name": det_name,
                **m,
                "metadata": {"checkpoint_dir": str(ckpt), "decision_threshold": threshold,
                             "target_fpr": TARGET_FPR},
            })
            print(f"  {test_method}: AUROC={m['auroc']:.4f}, Acc={m['accuracy']:.3f}")

        del sup_model, sup_tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

#Output results
unsupervised_results = [
    {"train_method": "score_only", "test_method": "ADAPTIVE_BINOCULARS",
     "detector_name": "Binoculars", **metrics_bino_bino},
    {"train_method": "score_only", "test_method": "ADAPTIVE_BINOCULARS",
     "detector_name": "FastDetectGPT", **metrics_bino_fdgpt},
    {"train_method": "score_only", "test_method": "ADAPTIVE_FASTDETECT",
     "detector_name": "Binoculars", **metrics_fdgpt_bino},
    {"train_method": "score_only", "test_method": "ADAPTIVE_FASTDETECT",
     "detector_name": "FastDetectGPT", **metrics_fdgpt_fdgpt},
]

all_results = {
    "unsupervised": unsupervised_results,
    "supervised": supervised_results,
    "method_distribution_binoculars": dict(counter_bino.most_common()),
    "method_distribution_fastdetect": dict(counter_fdgpt.most_common()),
    "num_test_prompts": n,
    "model": gen_model,
}

with open(OUTPUT_DIR / "adaptive_results.json", "w") as f:
    json.dump(all_results, f, indent=2)

# Save both selected corpora — test-split only
for tag, sels in [("binoculars", selections_bino), ("fastdetect", selections_fdgpt)]:
    with open(OUTPUT_DIR / f"adaptive_selected_{tag}.jsonl", "w") as f:
        for pid in test_pids:
            method, text, score = sels[pid]
            f.write(json.dumps({
                "prompt_id": pid, "text": text,
                "selected_method": method, "oracle_score": score,
            }) + "\n")


# Save ALL-split (train + val + test) as method-file JSONLs matching the
# existing generation schema, so the main training pipeline picks them up as
# new train_method rows on the next run.

gen_out_dir = OUTPUT_DIR / "generations" / "publication_detection"
gen_out_dir.mkdir(parents=True, exist_ok=True)

def _write_method_jsonl(out_path: Path, sels: dict, method_name: str) -> None:
    with open(out_path, "w") as fh:
        for pid in sorted(sels.keys()):
            method, text, score = sels[pid]
            row = {
                "prompt_id": pid,
                "method_name": method_name,
                "run_id": 0,
                "seed": 42,
                "text": text,
                "selected_method": method,
                "oracle_score": score,
            }
            fh.write(json.dumps(row) + "\n")
    print(f"  wrote {out_path} ({len(sels)} rows)")

_write_method_jsonl(gen_out_dir / "ADAPTIVE_BINOCULARS.jsonl", selections_bino, "ADAPTIVE_BINOCULARS")
_write_method_jsonl(gen_out_dir / "ADAPTIVE_FASTDETECT.jsonl", selections_fdgpt, "ADAPTIVE_FASTDETECT")

#Merge into main matrices and resave in OUTPUT_DIR
import csv
results_src = ARTIFACT_ROOT / "results" / "generative_detection" / "publication_detection"
if results_src.exists():
    print(f"\n=== Merging ADAPTIVE_* into main matrices from {results_src} ===")
    merged_dir = OUTPUT_DIR / "results" / "generative_detection" / "publication_detection"
    merged_dir.mkdir(parents=True, exist_ok=True)
    adaptive_methods = {"ADAPTIVE_SELECTION", "ADAPTIVE_BINOCULARS", "ADAPTIVE_FASTDETECT"}

    def _merge(name: str, new_rows: list[dict]) -> None:
        src = results_src / f"{name}.json"
        if not src.exists():
            print(f"  {name}.json: not found, skipping")
            return
        with open(src) as fh:
            rows = json.load(fh)
        rows = [r for r in rows if r.get("test_method") not in adaptive_methods]
        rows.extend(new_rows)
        with open(merged_dir / f"{name}.json", "w") as fh:
            json.dump(rows, fh, indent=2)
        fieldnames = sorted({k for r in rows for k in r if k != "metadata"})
        with open(merged_dir / f"{name}.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fieldnames)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k) for k in fieldnames})
        print(f"  {name}: +{len(new_rows)} rows (total {len(rows)})")

    _merge("matrix_score_only", unsupervised_results)
    _merge("matrix_unsupervised", unsupervised_results)
    _merge("matrix_supervised", supervised_results)
else:
    print(f"\nNo main results directory at {results_src} — skipping matrix merge.")

print(f"\n=== SUMMARY ===")
print(f"ADAPTIVE_BINOCULARS  | Binoc {metrics_bino_bino['auroc']:.4f}  FDGPT {metrics_bino_fdgpt['auroc']:.4f}")
print(f"ADAPTIVE_FASTDETECT  | Binoc {metrics_fdgpt_bino['auroc']:.4f}  FDGPT {metrics_fdgpt_fdgpt['auroc']:.4f}")
for r in supervised_results:
    print(f"{r['detector_name']:<18} ({r['train_method']:<20}) {r['test_method']:<20} AUROC: {r['auroc']:.4f}")
print(f"\nResults saved to {OUTPUT_DIR}")
