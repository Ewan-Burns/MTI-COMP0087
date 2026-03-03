import os
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import numpy as np

import mlx.core as mx
from huggingface_hub import snapshot_download
from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler

# Training stack
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


# -----------------------------
# User-provided function
# -----------------------------
# Must exist in your environment.
# It returns: [[human_text, prompt], ...]
from get_prompts import get_n_prompts  # <-- change this import


# -----------------------------
# Config
# -----------------------------
# MODEL_REPO_ID = os.environ.get(
#     "MODEL_REPO_ID",
#     "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
# )
MODEL_REPO_ID = os.environ.get(
    "MODEL_REPO_ID",
    "mlx-community/TinyLlama-1.1B-Chat-v1.0-4bit",
)

LOCAL_MODEL_PATH = Path(
    os.environ.get("LOCAL_MLX_MODEL_PATH", f"./models/{MODEL_REPO_ID.split('/')[-1]}")
).expanduser()
DOWNLOAD_IF_MISSING = os.environ.get("DOWNLOAD_IF_MISSING", "1").lower() not in ("0", "false", "no")
HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")

# Keep generation cheaper
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "128"))
SAMPLE_SEED = int(os.environ.get("SAMPLE_SEED", "42"))

# Prompts (default lowered for speed)
N_PROMPTS = int(os.environ.get("N_PROMPTS", "60"))  # set to 100 if fast enough

# RoBERTa training knobs
ROBERTA_MODEL_ID = os.environ.get("ROBERTA_MODEL_ID", "roberta-base")
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "./roberta_cross_eval_runs")).expanduser()
MAX_LENGTH = int(os.environ.get("MAX_LENGTH", "256"))
EPOCHS = float(os.environ.get("EPOCHS", "1"))
TRAIN_BATCH_SIZE = int(os.environ.get("TRAIN_BATCH_SIZE", "8"))
EVAL_BATCH_SIZE = int(os.environ.get("EVAL_BATCH_SIZE", "16"))
LEARNING_RATE = float(os.environ.get("LEARNING_RATE", "2e-5"))
WEIGHT_DECAY = float(os.environ.get("WEIGHT_DECAY", "0.01"))
WARMUP_RATIO = float(os.environ.get("WARMUP_RATIO", "0.0"))
TRAIN_SPLIT = float(os.environ.get("TRAIN_SPLIT", "0.8"))
VAL_SPLIT = float(os.environ.get("VAL_SPLIT", "0.1"))  # remainder is test
RUN_NAME = os.environ.get("RUN_NAME", "cross_generalization")

# Cache generations so reruns are fast
GEN_CACHE_PATH = Path(os.environ.get("GEN_CACHE_PATH", "./gen_cache.json")).expanduser()


# -----------------------------
# Utilities: ensure local MLX model
# -----------------------------
from pathlib import Path

def find_mlx_model_dir(root: Path) -> Path | None:
    """
    Return a directory that contains config.json and at least one .safetensors file.
    Searches root first, then recursively. If multiple candidates, pick the one
    with the largest total safetensors size.
    """
    root = root.resolve()
    candidates: list[Path] = []

    def is_candidate(p: Path) -> bool:
        if not p.is_dir():
            return False
        if not (p / "config.json").exists():
            return False
        # weights can be named many things; search just in this dir
        return any(p.glob("*.safetensors"))

    # 1) root itself
    if is_candidate(root):
        return root

    # 2) search recursively for directories containing config.json
    for cfg in root.rglob("config.json"):
        d = cfg.parent
        if is_candidate(d):
            candidates.append(d)

    if not candidates:
        return None

    # Pick the candidate with the largest sum of safetensors sizes (usually the real model)
    def total_weight_bytes(d: Path) -> int:
        return sum(f.stat().st_size for f in d.glob("*.safetensors"))

    candidates.sort(key=total_weight_bytes, reverse=True)
    return candidates[0]


def has_local_model_files(model_path: Path) -> bool:
    return find_mlx_model_dir(model_path) is not None


def ensure_local_model(model_path: Path, repo_id: str) -> Path:
    model_path = model_path.resolve()

    found = find_mlx_model_dir(model_path)
    if found is not None:
        return found

    if not DOWNLOAD_IF_MISSING:
        raise FileNotFoundError(
            f"Local model not found at: {model_path}\n"
            "Enable DOWNLOAD_IF_MISSING=1 or set LOCAL_MLX_MODEL_PATH to an existing local model."
        )

    model_path.mkdir(parents=True, exist_ok=True)
    print(f"Local model not found. Downloading {repo_id} to {model_path} ...")

    snapshot_download(
        repo_id=repo_id,
        local_dir=str(model_path),
        allow_patterns=[
            "*.json",
            "*.safetensors",  # IMPORTANT
            "*.py",
            "tokenizer.model",
            "*.tiktoken",
            "tiktoken.model",
            "*.txt",
            "*.jsonl",
            "*.jinja",
        ],
        token=HF_TOKEN,
    )

    found = find_mlx_model_dir(model_path)
    if found is None:
        raise RuntimeError(
            "Download completed but required MLX model files were not found.\n"
            f"Searched under: {model_path}\n"
            "Expected to find a directory containing config.json and *.safetensors."
        )

#     return found
# def has_local_model_files(model_path: Path) -> bool:
#     if not model_path.exists() or not model_path.is_dir():
#         return False
#     return (model_path / "config.json").exists() and any(model_path.glob("model*.safetensors"))

# def ensure_local_model(model_path: Path, repo_id: str) -> Path:
#     model_path = model_path.resolve()
#     if has_local_model_files(model_path):
#         return model_path

#     if not DOWNLOAD_IF_MISSING:
#         raise FileNotFoundError(
#             f"Local model not found at: {model_path}\n"
#             "Enable DOWNLOAD_IF_MISSING=1 or set LOCAL_MLX_MODEL_PATH to an existing local model."
#         )

#     model_path.mkdir(parents=True, exist_ok=True)
#     print(f"Local model not found. Downloading {repo_id} to {model_path} ...")

#     snapshot_download(
#         repo_id=repo_id,
#         local_dir=str(model_path),
#         allow_patterns=[
#             "*.json",
#             "model*.safetensors",
#             "*.safetensors", 
#             "*.py",
#             "tokenizer.model",
#             "*.tiktoken",
#             "tiktoken.model",
#             "*.txt",
#             "*.jsonl",
#             "*.jinja",
#         ],
#         token=HF_TOKEN,
#     )

#     if not has_local_model_files(model_path):
#         raise RuntimeError(f"Download completed but required model files were not found in {model_path}")
#     return model_path.resolve()


def load_model_and_tokenizer(model_path: Path) -> Tuple[Any, Any]:
    # Keep it simple; add your fix_mistral_regex back if you need it.
    return load(str(model_path))


# -----------------------------
# Logits processor: EXCLUDE_TOP_K
# -----------------------------
def make_exclude_top_k_processor(k: int):
    """
    Logits processor for mlx_lm.generate that removes the current top-k tokens
    by setting their logits to a very negative value.
    Uses only ops that exist in older mlx.core versions (no mx.scatter).
    """
    k = int(max(1, k))

    def _proc(tokens: mx.array, logits: mx.array) -> mx.array:
        # logits: [vocab]
        vocab = int(logits.shape[-1])
        kk = min(k, vocab)

        # indices of top-k largest logits
        top_idx = mx.argpartition(-logits, kth=kk - 1, axis=-1)[:kk]

        # build a boolean mask over vocab positions: True for tokens to exclude
        positions = mx.arange(vocab)
        # Compare each vocab position against each of the top_idx values
        # result shape: [vocab, kk] -> reduce any -> [vocab]
        exclude_mask = mx.any(positions[:, None] == top_idx[None, :], axis=1)

        # replace excluded logits with a large negative value
        neg_inf = mx.array(-1e9, dtype=logits.dtype)
        return mx.where(exclude_mask, neg_inf, logits)

    return _proc


# -----------------------------
# Regimes
# -----------------------------
@dataclass(frozen=True)
class Regime:
    name: str
    sampler: Any
    logits_processors: Any  # None or callable/list
    seed_base: int | None   # if None => deterministic / no reseed


def build_regimes() -> List[Regime]:
    # Greedy: deterministic
    greedy = Regime(
        name="GREEDY",
        sampler=make_sampler(temp=0.0),
        logits_processors=None,
        seed_base=None,
    )
    # Top-k
    topk10 = Regime(
        name="TOP_K_10",
        sampler=make_sampler(temp=0.8, top_k=10),
        logits_processors=None,
        seed_base=SAMPLE_SEED,
    )
    topk100 = Regime(
        name="TOP_K_100",
        sampler=make_sampler(temp=0.8, top_k=100),
        logits_processors=None,
        seed_base=SAMPLE_SEED,
    )
    # Exclude-top-k: still need sampling. Use temp + (optional) top_p to avoid weird tail.
    ex5 = Regime(
        name="EXCLUDE_TOP_K_5",
        sampler=make_sampler(temp=0.9, top_p=0.95),
        logits_processors=[make_exclude_top_k_processor(5)],
        seed_base=SAMPLE_SEED,
    )
    ex10 = Regime(
        name="EXCLUDE_TOP_K_10",
        sampler=make_sampler(temp=0.9, top_p=0.95),
        logits_processors=[make_exclude_top_k_processor(10)],
        seed_base=SAMPLE_SEED,
    )
    return [greedy, topk10, topk100, ex5, ex10]


# -----------------------------
# Generation + caching
# -----------------------------
def load_gen_cache(path: Path) -> Dict[str, Dict[str, str]]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}

def save_gen_cache(path: Path, cache: Dict[str, Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")

def prompt_to_tokens(tokenizer: Any, prompt: str):
    messages = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(messages, add_generation_prompt=True)

def generate_for_regime(
    model: Any,
    mlx_tokenizer: Any,
    prompts: List[str],
    regime: Regime,
    cache: Dict[str, Dict[str, str]],
) -> List[str]:
    """
    Returns generated texts aligned with prompts.
    Cache format: cache[regime.name][prompt] = generated_text
    """
    cache.setdefault(regime.name, {})

    outputs: List[str] = []
    for i, p in enumerate(prompts):
        if p in cache[regime.name]:
            outputs.append(cache[regime.name][p])
            continue

        if regime.seed_base is not None:
            # deterministic per (regime, index) to keep reproducible across runs
            mx.random.seed(regime.seed_base + i)

        tokens = prompt_to_tokens(mlx_tokenizer, p)
        out = generate(
            model,
            mlx_tokenizer,
            tokens,
            sampler=regime.sampler,
            logits_processors=regime.logits_processors,
            max_tokens=MAX_TOKENS,
        )
        cache[regime.name][p] = out
        outputs.append(out)

    return outputs


# -----------------------------
# Build datasets and splits
# -----------------------------
def make_binary_dataset(humans: List[str], ais: List[str], seed: int = 1234) -> Dataset:
    assert len(humans) == len(ais)
    texts = humans + ais
    labels = [0] * len(humans) + [1] * len(ais)

    # Shuffle together
    idx = list(range(len(texts)))
    rng = random.Random(seed)
    rng.shuffle(idx)
    texts = [texts[i] for i in idx]
    labels = [labels[i] for i in idx]

    return Dataset.from_dict({"text": texts, "label": labels})

def split_dataset(ds: Dataset, train_frac: float, val_frac: float, seed: int = 1234):
    assert 0 < train_frac < 1
    assert 0 <= val_frac < 1
    assert train_frac + val_frac < 1

    ds = ds.shuffle(seed=seed)
    n = len(ds)
    n_train = int(round(n * train_frac))
    n_val = int(round(n * val_frac))
    n_test = n - n_train - n_val
    if n_test <= 0:
        # ensure at least 1 test example
        n_test = 1
        if n_val > 0:
            n_val -= 1
        else:
            n_train -= 1

    train = ds.select(range(0, n_train))
    val = ds.select(range(n_train, n_train + n_val)) if n_val > 0 else None
    test = ds.select(range(n_train + n_val, n_train + n_val + n_test))
    return train, val, test

def tokenize_dataset(ds: Dataset, tok: Any) -> Dataset:
    def _tok(batch):
        return tok(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
        )
    return ds.map(_tok, batched=True)


# -----------------------------
# Metrics
# -----------------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = 1 / (1 + np.exp(-logits[:, 1])) if logits.shape[-1] == 2 else None
    preds = np.argmax(logits, axis=-1)

    out = {
        "acc": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds),
    }
    # ROC-AUC needs probabilities and both classes present
    try:
        if probs is not None and len(set(labels.tolist() if hasattr(labels, "tolist") else labels)) == 2:
            out["roc_auc"] = roc_auc_score(labels, probs)
    except Exception:
        pass
    return out
import inspect
from transformers import TrainingArguments

def make_training_args(output_dir: str, run_name: str, seed: int, learning_rate: float,
                       weight_decay: float, warmup_ratio: float, num_train_epochs: float,
                       per_device_train_batch_size: int, per_device_eval_batch_size: int,
                       evaluation_strategy: str | None = "epoch",
                       save_strategy: str | None = "no",
                       logging_steps: int = 20,
                       report_to: str | None = "none",
                       fp16: bool = True) -> TrainingArguments:
    """
    Build TrainingArguments in a way that's compatible with the installed transformers.
    Filters out keywords not accepted by the local TrainingArguments signature.
    """
    base_kwargs = dict(
        output_dir=str(output_dir),
        run_name=str(run_name),
        seed=seed,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        evaluation_strategy=evaluation_strategy,
        save_strategy=save_strategy,
        logging_steps=logging_steps,
        report_to=report_to,
        fp16=fp16,
    )

    # Remove None values (we only pass configured options)
    base_kwargs = {k: v for k, v in base_kwargs.items() if v is not None}

    # Filter to only parameters supported by this version of TrainingArguments
    sig = inspect.signature(TrainingArguments)
    supported_params = set(sig.parameters.keys())
    filtered_kwargs = {k: v for k, v in base_kwargs.items() if k in supported_params}

    # For older transformers that don't have evaluation_strategy but do have
    # `evaluate_during_training`/`do_eval` or similar, we could map them. Try a couple guesses:
    if "evaluation_strategy" not in filtered_kwargs:
        if "evaluate_during_training" in supported_params and evaluation_strategy is not None:
            # map epoch -> True (best-effort)
            filtered_kwargs["evaluate_during_training"] = evaluation_strategy in ("epoch", "always")
        if "do_eval" in supported_params:
            # keep eval on if user wanted epoch evaluation
            filtered_kwargs["do_eval"] = evaluation_strategy is not None

    return TrainingArguments(**filtered_kwargs)

# -----------------------------
# Train one RoBERTa model
# -----------------------------
def train_roberta_classifier(
    train_ds: Dataset,
    val_ds: Dataset | None,
    run_dir: Path,
    seed: int = 1234,
) -> Tuple[Any, Any]:
    tok = AutoTokenizer.from_pretrained(ROBERTA_MODEL_ID, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(ROBERTA_MODEL_ID, num_labels=2)

    train_tok = tokenize_dataset(train_ds, tok)
    val_tok = tokenize_dataset(val_ds, tok) if val_ds is not None else None

    # Choose fp16 if CUDA is available (trainer handles it)
    args = make_training_args(
        output_dir=run_dir,
        run_name=run_dir.name,
        seed=seed,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        evaluation_strategy="epoch" if val_tok is not None else None,
        save_strategy="no",
        logging_steps=20,
        report_to="none",
        fp16=True,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        compute_metrics=compute_metrics if val_tok is not None else None,
    )

    trainer.train()
    return trainer, tok


def eval_trainer_on_dataset(trainer: Trainer, tok: Any, ds: Dataset) -> Dict[str, float]:
    ds_tok = tokenize_dataset(ds, tok)
    preds = trainer.predict(ds_tok)
    metrics = compute_metrics((preds.predictions, preds.label_ids))
    return metrics


# -----------------------------
# Main
# -----------------------------
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1) Get prompt/human pairs
    pairs = get_n_prompts(N_PROMPTS)
    humans = [str(x[0]).strip() for x in pairs]
    prompts = [str(x[1]).strip() for x in pairs]
    assert len(humans) == len(prompts) == N_PROMPTS

    # 2) Load MLX model
    model_path = ensure_local_model(LOCAL_MODEL_PATH, MODEL_REPO_ID)
    model, mlx_tokenizer = load_model_and_tokenizer(model_path)

    regimes = build_regimes()
    print(f"Using MLX model: {model_path}")
    print(f"Prompts: {N_PROMPTS}")
    print(f"Max gen tokens: {MAX_TOKENS}")
    print("Regimes:", ", ".join(r.name for r in regimes))

    # 3) Load generation cache, generate missing
    cache = load_gen_cache(GEN_CACHE_PATH)

    ai_by_regime: Dict[str, List[str]] = {}
    for r in regimes:
        print(f"\nGenerating for regime: {r.name}")
        ai_texts = generate_for_regime(model, mlx_tokenizer, prompts, r, cache)
        ai_by_regime[r.name] = ai_texts
        save_gen_cache(GEN_CACHE_PATH, cache)
        print(f"Done: {r.name}")

    # 4) Build datasets per regime (same humans, different ai)
    datasets_by_regime = {}
    for r in regimes:
        ds = make_binary_dataset(humans, ai_by_regime[r.name], seed=1234)
        train, val, test = split_dataset(ds, TRAIN_SPLIT, VAL_SPLIT, seed=1234)
        datasets_by_regime[r.name] = {"train": train, "val": val, "test": test}
        print(f"{r.name}: train={len(train)}, val={len(val) if val is not None else 0}, test={len(test)}")

    # 5) Train one RoBERTa classifier per regime
    trainers = {}
    tokenizers = {}

    for r in regimes:
        run_dir = OUTPUT_DIR / f"{RUN_NAME}_train_{r.name}"
        run_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nTraining RoBERTa for regime: {r.name} -> {run_dir}")
        trainer, tok = train_roberta_classifier(
            datasets_by_regime[r.name]["train"],
            datasets_by_regime[r.name]["val"],
            run_dir,
            seed=1234,
        )
        trainers[r.name] = trainer
        tokenizers[r.name] = tok
        print(f"Finished training: {r.name}")

    # 6) Cross-evaluate: train-regime model tested on each regime’s test set
    print("\nCross-evaluation (rows=train regime, cols=test regime):")
    names = [r.name for r in regimes]

    # We'll print a compact matrix for accuracy and F1 (and roc_auc if available)
    results: Dict[Tuple[str, str], Dict[str, float]] = {}
    for train_name in names:
        for test_name in names:
            metrics = eval_trainer_on_dataset(
                trainers[train_name],
                tokenizers[train_name],
                datasets_by_regime[test_name]["test"],
            )
            results[(train_name, test_name)] = metrics

    def fmt(m: Dict[str, float], key: str) -> str:
        v = m.get(key, float("nan"))
        if v != v:  # nan
            return "  n/a"
        return f"{v:5.3f}"

    # Print Acc matrix
    print("\nACC:")
    header = " " * 18 + " ".join(f"{n:>16}" for n in names)
    print(header)
    for tr in names:
        row = [f"{tr:>18}"]
        for te in names:
            row.append(f"{fmt(results[(tr, te)], 'acc'):>16}")
        print("".join(row))

    # Print F1 matrix
    print("\nF1:")
    print(header)
    for tr in names:
        row = [f"{tr:>18}"]
        for te in names:
            row.append(f"{fmt(results[(tr, te)], 'f1'):>16}")
        print("".join(row))

    # Optional ROC-AUC matrix (may be n/a depending on logits->prob mapping)
    print("\nROC_AUC (if available):")
    print(header)
    for tr in names:
        row = [f"{tr:>18}"]
        for te in names:
            row.append(f"{fmt(results[(tr, te)], 'roc_auc'):>16}")
        print("".join(row))

    print(f"\nGeneration cache: {GEN_CACHE_PATH.resolve()}")
    print(f"Model runs stored under: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()