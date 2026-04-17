"""
Fast version of adaptive_selection.py

Usage (Colab):
    %run scripts/adaptive_selection_fastwrite.py \
        --artifact-root /content/drive/MyDrive/statnlp_artifacts \
        --output-dir /content/drive/MyDrive/adaptive_results_qwen
"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
import numpy as np
from tqdm import tqdm
from collections import Counter
from transformers import AutoModelForCausalLM, AutoTokenizer

#Parser setup
parser = argparse.ArgumentParser()
parser.add_argument("--artifact-root", required=True)
parser.add_argument("--output-dir", required=True)
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument("--max-length", type=int, default=512)
args = parser.parse_args()

ARTIFACT_ROOT = Path(args.artifact_root).expanduser().resolve()
OUTPUT_DIR = Path(args.output_dir).expanduser().resolve()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
BATCH_SIZE = args.batch_size
MAX_LENGTH = args.max_length
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16

#Load prompts, generations
ds_dir = ARTIFACT_ROOT / "datasets" / "generative_detection" / "publication_detection"
gen_dir = ARTIFACT_ROOT / "generations" / "publication_detection"
assert ds_dir.exists() and gen_dir.exists(), f"Missing: {ds_dir} or {gen_dir}"

#Read Prompts In
with open(ds_dir / "prompts.jsonl") as f:
    all_prompts = [json.loads(line) for line in f]
all_prompts_by_id = {p["prompt_id"]: p for p in all_prompts}
print(f"Prompts: {len(all_prompts)} total")

methods = sorted([p.stem for p in gen_dir.glob("*.jsonl")
                  if not p.stem.startswith(".") and not p.stem.startswith("ADAPTIVE_")])
print(f"Methods ({len(methods)}): {methods}")

prompt_rows: dict[str, dict[str, dict]] = {}
for method in methods:
    with open(gen_dir / f"{method}.jsonl") as f:
        for line in f:
            row = json.loads(line)
            prompt_rows.setdefault(row["prompt_id"], {})[method] = row

#Load main-reference pair
with open(gen_dir / "manifest.json") as f:
    gen_manifest = json.load(f)
gen_model = gen_manifest["publication_model_id"]
main_model = gen_model if "-Instruct" not in gen_model else gen_model.replace("-Instruct", "")
aux_model = gen_model + "-Instruct" if "-Instruct" not in gen_model else gen_model
print(f"LM pair: {main_model} (q) + {aux_model} (r)")

#Setup for HuggingFace Models
token = os.environ.get("HF_TOKEN")
tokenizer = AutoTokenizer.from_pretrained(main_model, token=token)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model_q = AutoModelForCausalLM.from_pretrained(main_model, torch_dtype=DTYPE, token=token).to(DEVICE).eval()
model_r = AutoModelForCausalLM.from_pretrained(aux_model, torch_dtype=DTYPE, token=token).to(DEVICE).eval()


def score_both_batch(texts: list[str], desc: str = "scoring") -> tuple[list[float], list[float]]:
    '''
    Function that returns both fast binoculars and FastDetectGPT
    '''
    bino, fdg = [], []
    for start in tqdm(range(0, len(texts), BATCH_SIZE), desc=desc, unit="batch", leave=False):
        batch = texts[start:start+BATCH_SIZE]
        enc = tokenizer(batch, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
        enc = {k: v.to(DEVICE) for k, v in enc.items()}

        with torch.no_grad():
            lq = model_q(**enc).logits[:, :-1, :]
            lr = model_r(**enc).logits[:, :-1, :]
        
        lab = enc["input_ids"][:, 1:]
        mask = enc["attention_mask"][:, 1:].float()

        lp = torch.log_softmax(lq, dim=-1)
        pr = torch.softmax(lr, dim=-1)

        sl = mask.sum(dim=-1).clamp(min=1.0)
        nll = -torch.gather(lp, dim=-1, index=lab.unsqueeze(-1)).squeeze(-1)
        ce = -(pr * lp).sum(dim=-1)
        bino.extend((((nll*mask).sum(-1)/sl) / ((ce*mask).sum(-1)/sl).clamp(min=1e-8)).cpu().tolist())
        actual = torch.gather(lp, dim=-1, index=lab.unsqueeze(-1)).squeeze(-1)
        expected = (pr * lp).sum(dim=-1)
        var = ((pr * lp**2).sum(dim=-1) - expected**2).clamp(min=1e-8)
        num = ((actual - expected) * mask).sum(dim=-1)
        fdg.extend((num / (var * mask).sum(dim=-1).clamp(min=1e-8).sqrt()).cpu().tolist())
    return bino, fdg


#Score all texts per method
print("\n=== Scoring all texts ===")
prompt_bino, prompt_fdgpt = {}, {}
for method in methods:
    texts, pids = [], []
    for pid in sorted(prompt_rows.keys()):
        if method in prompt_rows[pid]:
            texts.append(prompt_rows[pid][method]["text"])
            pids.append(pid)
    if not texts:
        continue
    b, f = score_both_batch(texts, desc=f"  {method}")
    for pid, bs, fs in zip(pids, b, f):
        prompt_bino.setdefault(pid, {})[method] = bs
        prompt_fdgpt.setdefault(pid, {})[method] = fs
    print(f"  {method}: bino={np.mean(b):.4f}, fdgpt={np.mean(f):.4f}")


def select(prompt_scores: dict, direction: str) -> tuple[dict, Counter]:
    '''
    Function that selects best method according to specific direction: max for binoculars, min for fastdetect
    '''
    picker = max if direction == "max" else min
    cnt, sel = Counter(), {}
    for pid in sorted(prompt_rows.keys()):
        sc = prompt_scores.get(pid, {})
        if not sc:
            continue
        best = picker(sc, key=sc.get)
        cnt[best] += 1
        sel[pid] = (best, prompt_rows[pid][best]["text"], sc[best])
    return sel, cnt


selections_bino, counter_bino = select(prompt_bino, "max")
selections_fdgpt, counter_fdgpt = select(prompt_fdgpt, "min")

print("\nBinoculars oracle top picks:")
for m, c in counter_bino.most_common(8):
    print(f"  {m:<25} {c}")
print("FastDetectGPT oracle top picks:")
for m, c in counter_fdgpt.most_common(8):
    print(f"  {m:<25} {c}")

# Write method-file JSONLs (all splits)
gen_out_dir = OUTPUT_DIR / "generations" / "publication_detection"
gen_out_dir.mkdir(parents=True, exist_ok=True)

def write_method_jsonl(out_path: Path, sels: dict, method_name: str) -> None:
    '''
    Write JsonL files to directory
    '''
    with open(out_path, "w") as f:
        for pid in sorted(sels.keys()):
            sel_method, text, score = sels[pid]
            f.write(json.dumps({
                "prompt_id": pid,
                "method_name": method_name,
                "run_id": 0,
                "seed": 42,
                "text": text,
                "selected_method": sel_method,
                "oracle_score": score,
            }) + "\n")
    print(f"  wrote {out_path} ({len(sels)} rows)")

write_method_jsonl(gen_out_dir / "ADAPTIVE_BINOCULARS.jsonl", selections_bino, "ADAPTIVE_BINOCULARS")
write_method_jsonl(gen_out_dir / "ADAPTIVE_FASTDETECT.jsonl", selections_fdgpt, "ADAPTIVE_FASTDETECT")

print(f"\Finished writing files to {gen_out_dir}")
