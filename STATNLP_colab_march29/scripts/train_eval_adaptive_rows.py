"""Train + evaluate ADAPTIVE_BINOCULARS / ADAPTIVE_FASTDETECT as new train-method rows.

Assumes adaptive_selection.py has already been run and its output JSONLs
(ADAPTIVE_BINOCULARS.jsonl, ADAPTIVE_FASTDETECT.jsonl) have been copied into
the main artifact's generations dir.

Bypasses train_supervised_detectors (which would retrigger mixture retrain).
Trains only the 6 new checkpoints (3 archs x 2 adaptive methods), then
evaluates each new checkpoint against every test method, and appends new
rows to matrix_supervised.json.

Usage (Colab):
    %env HF_TOKEN=...
    %run scripts/train_eval_adaptive_rows.py \
        --artifact-root /content/drive/MyDrive/statnlp_artifacts
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Allow running from repo root without installing the package.
try:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
except NameError:
    pass
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

parser = argparse.ArgumentParser()
parser.add_argument("--artifact-root", required=True,
                    help="Existing artifacts directory (read-only for this script)")
parser.add_argument("--output-root", required=True,
                    help="Separate directory for all writes (new checkpoints, corpora, merged matrix)")
parser.add_argument("--dataset-name", default="publication_detection")
parser.add_argument("--adaptive-methods", nargs="+",
                    default=["ADAPTIVE_BINOCULARS", "ADAPTIVE_FASTDETECT"])
parser.add_argument("--extra-gen-dir",
                    help="Optional directory holding extra method JSONLs (e.g. ADAPTIVE_*.jsonl "
                         "from adaptive_selection.py output). Files here override artifact-root.")
parser.add_argument("--extra-matrix-src", nargs="*", default=[],
                    help="Additional matrix_supervised.json paths to merge in "
                         "(e.g. the adaptive test-column matrix from a previous run). "
                         "New rows from this run override any duplicates.")
args = parser.parse_args()

ARTIFACT_ROOT = Path(args.artifact_root).expanduser().resolve()
OUTPUT_ROOT = Path(args.output_root).expanduser().resolve()
EXTRA_GEN_DIR = Path(args.extra_gen_dir).expanduser().resolve() if args.extra_gen_dir else None
DATASET = args.dataset_name
ADAPTIVE_METHODS = args.adaptive_methods

# --- Paths ---
# READ from ARTIFACT_ROOT (untouched):
gen_dir = ARTIFACT_ROOT / "generations" / DATASET
ds_dir = ARTIFACT_ROOT / "datasets" / "generative_detection" / DATASET
existing_matrix_path = ARTIFACT_ROOT / "results" / "generative_detection" / DATASET / "matrix_supervised.json"
# WRITE to OUTPUT_ROOT (new dir, nothing clobbered):
models_dir = OUTPUT_ROOT / "models" / "detectors"
corpora_dir = OUTPUT_ROOT / "corpora" / DATASET
results_path = OUTPUT_ROOT / "results" / "generative_detection" / DATASET / "matrix_supervised.json"
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
print(f"READ from:       {ARTIFACT_ROOT}")
if EXTRA_GEN_DIR:
    print(f"EXTRA gen dir:   {EXTRA_GEN_DIR}")
print(f"WRITE to:        {OUTPUT_ROOT}")

for p in [gen_dir, ds_dir]:
    assert p.exists(), f"Missing: {p}"

# Build generation_paths: start from ARTIFACT_ROOT's gen dir, overlay anything in EXTRA_GEN_DIR.
generation_paths = {
    p.stem: p for p in sorted(gen_dir.glob("*.jsonl"))
    if not p.stem.startswith(".") and p.stem != "manifest"
}
if EXTRA_GEN_DIR and EXTRA_GEN_DIR.exists():
    for p in sorted(EXTRA_GEN_DIR.glob("*.jsonl")):
        if p.stem.startswith(".") or p.stem == "manifest":
            continue
        generation_paths[p.stem] = p  # override / add
print(f"Methods in generation_paths ({len(generation_paths)}): {list(generation_paths)}")

for m in ADAPTIVE_METHODS:
    assert m in generation_paths, (
        f"Missing adaptive method '{m}'. Looked in:\n"
        f"  artifact-root gen dir: {gen_dir}\n"
        f"  extra-gen-dir:         {EXTRA_GEN_DIR}"
    )

# --- Load config + manifest ---
from statnlp_bench.config import SupervisedTrainingConfig
from statnlp_bench.types import DatasetManifest
from statnlp_bench.tracks.generative_detection import build_detection_corpus
from statnlp_bench.detectors.supervised import load_detection_examples
from statnlp_bench.tracks._detection_scoring import _evaluate_supervised_cell, _evaluation_texts_for_target
from statnlp_bench.training.train_supervised import train_supervised_detector

prompts_records_path = ds_dir / "prompts.jsonl"
prompts_manifest_meta = {}
mf_json = ds_dir / "manifest.json"
if mf_json.exists():
    prompts_manifest_meta = json.loads(mf_json.read_text())

prompt_manifest = DatasetManifest(
    name=DATASET,
    track="generative_detection",
    external=bool(prompts_manifest_meta.get("external", False)),
    root_dir=ds_dir,
    records_path=prompts_records_path,
    metadata=prompts_manifest_meta,
)

training_config = SupervisedTrainingConfig()
print(f"Training config: archs={list(training_config.architecture_model_ids)}")

# --- Prep: assemble all rows up-front and save incrementally as we go ---
import gc
try:
    import torch
except ImportError:
    torch = None

def _free_mem():
    gc.collect()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()

results_path.parent.mkdir(parents=True, exist_ok=True)
all_rows: list[dict] = []
if existing_matrix_path.exists():
    all_rows = json.loads(existing_matrix_path.read_text())
    print(f"Loaded {len(all_rows)} rows from {existing_matrix_path}")
for extra in args.extra_matrix_src:
    p = Path(extra).expanduser().resolve()
    if not p.exists():
        print(f"--extra-matrix-src not found, skipping: {p}"); continue
    extra_rows = json.loads(p.read_text())
    by_key = {(r["detector_name"], r["train_method"], r["test_method"]): r for r in all_rows}
    for r in extra_rows:
        by_key[(r["detector_name"], r["train_method"], r["test_method"])] = r
    all_rows = list(by_key.values())
    print(f"Merged {len(extra_rows)} rows from {p} (total {len(all_rows)})")

def _save_matrix(rows: list[dict]) -> None:
    tmp = results_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(rows, indent=2))
    os.replace(tmp, results_path)

# --- Train 6 new checkpoints (durable: each saved to disk on completion) ---
new_checkpoints: dict[tuple[str, str], Path] = {}
for adapt_method in ADAPTIVE_METHODS:
    print(f"\n=== Building corpus for {adapt_method} ===")
    corpus_path = build_detection_corpus(
        prompt_manifest, generation_paths,
        source_methods=[adapt_method],
        output_dir=corpora_dir / adapt_method,
    )
    examples = load_detection_examples(corpus_path)
    train_ex = [e for e in examples if e.split == "train"]
    val_ex = [e for e in examples if e.split == "validation"]
    if not val_ex and train_ex:
        n_human = sum(1 for e in train_ex if e.label == 0)
        n_ai = sum(1 for e in train_ex if e.label == 1)
        ratio = training_config.internal_validation_ratio
        holdout_human = max(1, int(n_human * ratio))
        holdout_ai = max(1, int(n_ai * ratio))
        humans = [e for e in train_ex if e.label == 0][:holdout_human]
        ais = [e for e in train_ex if e.label == 1][:holdout_ai]
        val_ex = humans + ais
        val_ids = {e.example_id for e in val_ex}
        train_ex = [e for e in train_ex if e.example_id not in val_ids]
    print(f"  train={len(train_ex)} val={len(val_ex)}")

    for arch, model_id in training_config.architecture_model_ids.items():
        out_dir = models_dir / f"tuned-{arch}" / adapt_method
        print(f"\n--- Training {arch} on {adapt_method} -> {out_dir} ---")
        ref = train_supervised_detector(
            train_examples=train_ex,
            validation_examples=val_ex,
            model_name_or_path=model_id,
            output_dir=out_dir,
            config=training_config,
        )
        new_checkpoints[(arch, adapt_method)] = ref.checkpoint_dir
        _free_mem()
        print(f"  done: {ref.checkpoint_dir}")

# --- Evaluate each new checkpoint against every test method; save after each checkpoint ---
print(f"\n=== Evaluating {len(new_checkpoints)} checkpoints x {len(generation_paths)} test methods ===")
eval_cache = {t: _evaluation_texts_for_target(prompt_manifest, p) for t, p in generation_paths.items()}
key_index = {(r["detector_name"], r["train_method"], r["test_method"]): i for i, r in enumerate(all_rows)}

for (arch, train_method), ckpt in new_checkpoints.items():
    for test_method in sorted(generation_paths):
        human_texts, ai_texts, labels = eval_cache[test_method]
        cell = _evaluate_supervised_cell(
            human_texts=human_texts, ai_texts=ai_texts, labels=labels,
            train_method=train_method, target_method=test_method,
            architecture=arch, checkpoint_dir=ckpt,
            training_config=training_config,
        )
        row = {
            "train_method": cell.train_method, "test_method": cell.test_method,
            "detector_name": cell.detector_name, "auroc": cell.auroc,
            "accuracy": cell.accuracy, "f1": cell.f1,
            "mean_ai_prob": cell.mean_ai_prob, "threshold": cell.threshold,
            "fpr": cell.fpr, "tpr": cell.tpr, "metadata": cell.metadata,
        }
        k = (row["detector_name"], row["train_method"], row["test_method"])
        if k in key_index:
            all_rows[key_index[k]] = row
        else:
            key_index[k] = len(all_rows)
            all_rows.append(row)
        print(f"  [{arch} / {train_method}] {test_method}: AUROC={cell.auroc:.4f}")
    # Save after each (arch, train_method) row — ~21 cells at a time — durable against crashes
    _save_matrix(all_rows)
    _free_mem()
    print(f"  saved checkpoint: {results_path} ({len(all_rows)} total rows)")

print(f"\nFinal matrix at {results_path} ({len(all_rows)} total rows)")
