import os
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import mlx.core as mx
from huggingface_hub import snapshot_download
from mlx_lm import generate, load
from mlx_lm.sample_utils import make_logits_processors, make_sampler

try:
    from transformers import pipeline
except ImportError:
    pipeline = None

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

# Model repo and local cache folder.
MODEL_REPO_ID = os.environ.get(
    "MODEL_REPO_ID",
    "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
)
LOCAL_MODEL_PATH = Path(
    os.environ.get("LOCAL_MLX_MODEL_PATH", f"./models/{MODEL_REPO_ID.split('/')[-1]}")
).expanduser()
DOWNLOAD_IF_MISSING = os.environ.get("DOWNLOAD_IF_MISSING", "1").lower() not in (
    "0",
    "false",
    "no",
)
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "256"))
SAMPLE_SEED = int(os.environ.get("SAMPLE_SEED", "42"))
BEAM_WIDTH = int(os.environ.get("BEAM_WIDTH", "4"))
BEAM_LENGTH_PENALTY = float(os.environ.get("BEAM_LENGTH_PENALTY", "0.6"))
ENABLE_AI_DETECTOR = os.environ.get("ENABLE_AI_DETECTOR", "1").lower() not in (
    "0",
    "false",
    "no",
)
DEFAULT_AI_DETECTOR_MODEL_IDS = [
    "openai-community/roberta-base-openai-detector",
    "Hello-SimpleAI/chatgpt-detector-roberta",
    "Hello-SimpleAI/chatgpt-qa-detector-roberta",
    "vraj33/ai-text-detector-deberta",
]
AI_DETECTOR_MODEL_ID = os.environ.get("AI_DETECTOR_MODEL_ID", "").strip()
AI_DETECTOR_MODEL_IDS_ENV = os.environ.get("AI_DETECTOR_MODEL_IDS", "").strip()
if AI_DETECTOR_MODEL_IDS_ENV:
    AI_DETECTOR_MODEL_IDS = [
        model_id.strip()
        for model_id in AI_DETECTOR_MODEL_IDS_ENV.split(",")
        if model_id.strip()
    ]
elif AI_DETECTOR_MODEL_ID:
    AI_DETECTOR_MODEL_IDS = [AI_DETECTOR_MODEL_ID]
else:
    AI_DETECTOR_MODEL_IDS = list(DEFAULT_AI_DETECTOR_MODEL_IDS)
AI_DETECTOR_CACHE_DIR = Path(
    os.environ.get("AI_DETECTOR_CACHE_DIR", "./models/detectors")
).expanduser()
AI_DETECTOR_MAX_CHARS = int(os.environ.get("AI_DETECTOR_MAX_CHARS", "2500"))
FIX_MISTRAL_REGEX = os.environ.get("FIX_MISTRAL_REGEX", "1").lower() not in (
    "0",
    "false",
    "no",
)
DEFAULT_PROMPT_TEXT = (
    "I am a GP, and I have a clinic in the morning, "
    "what's my first patient most likely to be presenting with?"
)
PROMPTS_ENV = os.environ.get("PROMPTS", DEFAULT_PROMPT_TEXT).strip()
PROMPT_SOURCE = os.environ.get("PROMPT_SOURCE", "mt_bench").strip().lower()
MT_BENCH_DATASET_REPO_ID = os.environ.get(
    "MT_BENCH_DATASET_REPO_ID", "HuggingFaceH4/mt_bench_prompts"
).strip()
MT_BENCH_DATASET_DIR = Path(
    os.environ.get("MT_BENCH_DATASET_DIR", "./datasets/mt_bench_prompts")
).expanduser()
MT_BENCH_MAX_PROMPTS = max(1, int(os.environ.get("MT_BENCH_MAX_PROMPTS", "16")))
MT_BENCH_BALANCE_CATEGORIES = os.environ.get(
    "MT_BENCH_BALANCE_CATEGORIES", "1"
).lower() not in ("0", "false", "no")
SAMPLING_RUNS_PER_PROMPT = max(
    1,
    int(os.environ.get("SAMPLING_RUNS_PER_PROMPT", "3")),
)
PROMPT_SEED_STRIDE = int(os.environ.get("PROMPT_SEED_STRIDE", "1000"))
ENABLE_HUMAN_BASELINE = os.environ.get("ENABLE_HUMAN_BASELINE", "1").lower() not in (
    "0",
    "false",
    "no",
)
HUMAN_BASELINE_TEXTS = [
    text.strip()
    for text in os.environ.get(
        "HUMAN_BASELINE_TEXTS",
        DEFAULT_PROMPT_TEXT,
    ).split("||")
    if text.strip()
]
HF_TOKEN = (
    os.environ.get("HF_TOKEN")
    or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    or os.environ.get("HUGGINGFACE_HUB_TOKEN")
)
PRINT_GENERATED_TEXT = os.environ.get("PRINT_GENERATED_TEXT", "0").lower() not in (
    "0",
    "false",
    "no",
)


def has_local_model_files(model_path: Path) -> bool:
    if not model_path.exists() or not model_path.is_dir():
        return False
    return (model_path / "config.json").exists() and any(
        model_path.glob("model*.safetensors")
    )


def ensure_local_model(model_path: Path, repo_id: str) -> Path:
    model_path = model_path.resolve()
    if has_local_model_files(model_path):
        return model_path

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
            "model*.safetensors",
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

    if not has_local_model_files(model_path):
        raise RuntimeError(
            f"Download completed but required model files were not found in {model_path}"
        )
    return model_path.resolve()


def _top_k_logprobs(logprobs: mx.array, k: int) -> tuple[list[int], list[float]]:
    vocab_size = int(logprobs.shape[-1])
    k = max(1, min(k, vocab_size))

    candidate_idx = mx.argpartition(-logprobs, kth=k - 1, axis=-1)[:k]
    candidate_logprobs = logprobs[candidate_idx]

    order = mx.argsort(-candidate_logprobs, axis=-1)
    top_idx = mx.take_along_axis(candidate_idx, order, axis=-1)
    top_logprobs = mx.take_along_axis(candidate_logprobs, order, axis=-1)

    return [int(x) for x in top_idx.tolist()], [float(x) for x in top_logprobs.tolist()]


def beam_search_generate(
    model: Any,
    tokenizer: Any,
    prompt: list[int] | mx.array | str,
    *,
    max_tokens: int,
    beam_width: int,
    length_penalty: float,
) -> str:
    if isinstance(prompt, str):
        add_special_tokens = tokenizer.bos_token is None or not prompt.startswith(
            tokenizer.bos_token
        )
        prompt_tokens = tokenizer.encode(prompt, add_special_tokens=add_special_tokens)
    elif isinstance(prompt, mx.array):
        prompt_tokens = [int(t) for t in prompt.tolist()]
    else:
        prompt_tokens = [int(t) for t in prompt]

    prompt_len = len(prompt_tokens)
    eos_token_ids = {int(t) for t in (getattr(tokenizer, "eos_token_ids", []) or [])}

    beams = [{"tokens": prompt_tokens, "logprob": 0.0, "finished": False}]

    def beam_score(beam: dict[str, Any]) -> float:
        generated_len = max(1, len(beam["tokens"]) - prompt_len)
        norm = ((5.0 + generated_len) / 6.0) ** max(0.0, length_penalty)
        return beam["logprob"] / norm

    for _ in range(max_tokens):
        candidates = []
        for beam in beams:
            if beam["finished"]:
                candidates.append(beam)
                continue

            beam_tokens = mx.array(beam["tokens"], dtype=mx.uint32)
            logits = model(beam_tokens[None])[:, -1, :].squeeze(0)
            logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)

            next_token_ids, next_token_logprobs = _top_k_logprobs(logprobs, beam_width)
            for token_id, token_logprob in zip(next_token_ids, next_token_logprobs):
                new_tokens = beam["tokens"] + [token_id]
                candidates.append(
                    {
                        "tokens": new_tokens,
                        "logprob": beam["logprob"] + token_logprob,
                        "finished": token_id in eos_token_ids,
                    }
                )

        candidates.sort(key=beam_score, reverse=True)
        beams = candidates[:beam_width]
        if all(b["finished"] for b in beams):
            break

    best_beam = max(beams, key=beam_score)
    completion_tokens = best_beam["tokens"][prompt_len:]
    if completion_tokens and completion_tokens[-1] in eos_token_ids:
        completion_tokens = completion_tokens[:-1]
    return tokenizer.decode(completion_tokens)


def load_ai_detector(model_id: str, token: str | None = None) -> Any:
    if pipeline is None:
        return None
    try:
        return pipeline(
            "text-classification",
            model=model_id,
            tokenizer=model_id,
            truncation=True,
            token=token,
        )
    except TypeError:
        # Compatibility fallback for older Transformers versions.
        try:
            return pipeline(
                "text-classification",
                model=model_id,
                tokenizer=model_id,
                truncation=True,
                use_auth_token=token,
            )
        except Exception as exc:
            print(f"AI detector unavailable ({model_id}): {exc}")
            return None
    except Exception as exc:
        print(f"AI detector unavailable ({model_id}): {exc}")
        return None


def load_model_and_tokenizer(model_path: Path, fix_mistral_regex: bool) -> tuple[Any, Any]:
    load_kwargs: dict[str, Any] = {}
    if fix_mistral_regex:
        load_kwargs["tokenizer_config"] = {"fix_mistral_regex": True}

    try:
        return load(str(model_path), **load_kwargs)
    except TypeError:
        # Older mlx_lm versions may not support tokenizer_config.
        if fix_mistral_regex:
            print(
                "mlx_lm.load does not accept tokenizer_config; retrying without fix_mistral_regex."
            )
        return load(str(model_path))


def iter_with_progress(items: list[Any], *, desc: str, unit: str) -> Any:
    total = len(items)
    if tqdm is not None:
        return tqdm(items, total=total, desc=desc, unit=unit, dynamic_ncols=True)

    def _simple_iterator() -> Any:
        for idx, item in enumerate(items, start=1):
            print(f"{desc}: {idx}/{total} {unit}")
            yield item

    return _simple_iterator()


def ensure_mt_bench_prompt_file(
    dataset_dir: Path, repo_id: str, token: str | None = None
) -> Path | None:
    question_file = (dataset_dir / "raw" / "question.jsonl").resolve()
    if question_file.exists():
        return question_file

    if not DOWNLOAD_IF_MISSING:
        print(
            "MT-Bench prompts not found locally and DOWNLOAD_IF_MISSING=0; skipping MT-Bench prompt source."
        )
        return None

    dataset_dir.mkdir(parents=True, exist_ok=True)
    print(f"MT-Bench prompts not found. Downloading {repo_id} to {dataset_dir} ...")
    try:
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=str(dataset_dir),
            allow_patterns=["raw/question.jsonl"],
            token=token,
        )
    except Exception as exc:
        print(f"MT-Bench prompt download failed ({repo_id}): {exc}")
        return None

    if not question_file.exists():
        print(f"MT-Bench prompt file missing after download: {question_file}")
        return None
    return question_file


def load_mt_bench_prompts(
    question_file: Path, *, max_prompts: int, balance_categories: bool
) -> list[str]:
    prompt_entries: list[dict[str, str]] = []
    with question_file.open("r", encoding="utf-8") as infile:
        for raw_line in infile:
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue

            turns = payload.get("turns") or []
            if not turns:
                continue
            prompt_text = str(turns[0]).strip()
            if not prompt_text:
                continue
            prompt_entries.append(
                {
                    "category": str(payload.get("category", "unknown")).strip() or "unknown",
                    "prompt": prompt_text,
                }
            )

    if not prompt_entries:
        return []

    if max_prompts >= len(prompt_entries):
        return [item["prompt"] for item in prompt_entries]

    if not balance_categories:
        return [item["prompt"] for item in prompt_entries[:max_prompts]]

    by_category: dict[str, list[str]] = defaultdict(list)
    for item in prompt_entries:
        by_category[item["category"]].append(item["prompt"])

    selected: list[str] = []
    categories = sorted(by_category.keys())
    row_idx = 0
    while len(selected) < max_prompts:
        added = False
        for category in categories:
            prompts = by_category[category]
            if row_idx < len(prompts):
                selected.append(prompts[row_idx])
                added = True
                if len(selected) >= max_prompts:
                    break
        if not added:
            break
        row_idx += 1
    return selected


def resolve_prompts() -> tuple[list[str], str]:
    if PROMPTS_ENV:
        prompt_list = [text.strip() for text in PROMPTS_ENV.split("||") if text.strip()]
        if prompt_list:
            return prompt_list, "PROMPTS env override"

    if PROMPT_SOURCE in ("mt_bench", "mt-bench", "mtbench"):
        question_file = ensure_mt_bench_prompt_file(
            MT_BENCH_DATASET_DIR, MT_BENCH_DATASET_REPO_ID, token=HF_TOKEN
        )
        if question_file is not None:
            prompt_list = load_mt_bench_prompts(
                question_file,
                max_prompts=MT_BENCH_MAX_PROMPTS,
                balance_categories=MT_BENCH_BALANCE_CATEGORIES,
            )
            if prompt_list:
                return prompt_list, "MT-Bench first-turn prompts"
        print("Falling back to built-in prompt due to unavailable MT-Bench prompts.")

    return ["Tell me a story about a dragon."], "Built-in fallback prompt"


def has_local_detector_files(detector_path: Path) -> bool:
    if not detector_path.exists() or not detector_path.is_dir():
        return False
    return (detector_path / "config.json").exists() and (
        any(detector_path.glob("*.safetensors"))
        or any(detector_path.glob("pytorch_model*.bin"))
        or any(detector_path.glob("*.bin"))
    )


def ensure_local_detector(
    cache_dir: Path, repo_id: str, token: str | None = None
) -> Path | None:
    detector_path = (cache_dir / repo_id.replace("/", "--")).resolve()
    if has_local_detector_files(detector_path):
        return detector_path

    if not DOWNLOAD_IF_MISSING:
        print(
            f"Detector model not found at {detector_path} and DOWNLOAD_IF_MISSING=0; skipping {repo_id}."
        )
        return None

    detector_path.mkdir(parents=True, exist_ok=True)
    print(f"Detector model not found. Downloading {repo_id} to {detector_path} ...")
    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(detector_path),
            allow_patterns=[
                "*.json",
                "*.safetensors",
                "*.bin",
                "*.txt",
                "*.model",
                "tokenizer*",
                "vocab*",
                "merges.txt",
            ],
            token=token,
        )
    except Exception as exc:
        print(f"Detector download failed ({repo_id}): {exc}")
        return None

    if not has_local_detector_files(detector_path):
        print(
            f"Detector download completed but required files were not found in {detector_path}"
        )
        return None
    return detector_path


def load_ai_detectors(
    model_ids: list[str], cache_dir: Path, token: str | None = None
) -> list[dict[str, Any]]:
    loaded_detectors: list[dict[str, Any]] = []
    for model_id in model_ids:
        local_override = Path(model_id).expanduser()
        if local_override.exists():
            model_ref = str(local_override.resolve())
        else:
            local_snapshot = ensure_local_detector(cache_dir, model_id, token=token)
            if local_snapshot is None:
                continue
            model_ref = str(local_snapshot)

        detector = load_ai_detector(model_ref, token=token)
        if detector is not None:
            ai_labels, human_labels, label_hint = infer_detector_label_sets(detector)
            loaded_detectors.append(
                {
                    "name": model_id,
                    "detector": detector,
                    "ai_labels": ai_labels,
                    "human_labels": human_labels,
                    "label_hint": label_hint,
                }
            )
    return loaded_detectors


def _normalize_label(label: str) -> str:
    return re.sub(r"\s+", " ", str(label).strip().lower())


def _classify_label_text(label_text: str) -> str:
    normalized = _normalize_label(label_text)
    ai_markers = ("ai", "fake", "generated", "machine", "chatgpt", "gpt")
    human_markers = ("human", "real", "organic", "person")

    if any(marker in normalized for marker in ai_markers):
        return "ai"
    if any(marker in normalized for marker in human_markers):
        return "human"
    return "unknown"


def infer_detector_label_sets(detector: Any) -> tuple[set[str], set[str], str]:
    ai_labels: set[str] = set()
    human_labels: set[str] = set()

    config = getattr(getattr(detector, "model", None), "config", None)
    id2label = getattr(config, "id2label", None)

    label_by_id: dict[int, str] = {}
    if isinstance(id2label, dict):
        for idx, label in id2label.items():
            try:
                label_by_id[int(idx)] = str(label)
            except (TypeError, ValueError):
                continue

    ai_ids: set[int] = set()
    human_ids: set[int] = set()
    for idx, label_text in label_by_id.items():
        label_class = _classify_label_text(label_text)
        if label_class == "ai":
            ai_ids.add(idx)
        elif label_class == "human":
            human_ids.add(idx)

    if not ai_ids and human_ids and len(label_by_id) == 2:
        ai_ids = set(label_by_id.keys()) - human_ids

    for idx in ai_ids:
        ai_labels.add(_normalize_label(f"LABEL_{idx}"))
        if idx in label_by_id:
            ai_labels.add(_normalize_label(label_by_id[idx]))
    for idx in human_ids:
        human_labels.add(_normalize_label(f"LABEL_{idx}"))
        if idx in label_by_id:
            human_labels.add(_normalize_label(label_by_id[idx]))

    if ai_labels or human_labels:
        return ai_labels, human_labels, "model-config labels"
    return set(), set(), "heuristic labels"


def _extract_ai_probability(
    scores: Any,
    *,
    ai_labels: set[str],
    human_labels: set[str],
) -> float | None:
    if isinstance(scores, dict):
        entries = [scores]
    elif isinstance(scores, list):
        entries = scores
    else:
        return None

    label_scores: list[tuple[str, float]] = []
    for item in entries:
        if not isinstance(item, dict):
            continue
        label = _normalize_label(item.get("label", ""))
        score = float(item.get("score", 0.0))
        label_scores.append((label, max(0.0, min(1.0, score))))

    if ai_labels:
        ai_probs = [score for label, score in label_scores if label in ai_labels]
        if ai_probs:
            return max(ai_probs)

    if human_labels:
        human_probs = [score for label, score in label_scores if label in human_labels]
        if human_probs:
            return 1.0 - max(human_probs)

    for label, score in label_scores:
        if "fake" in label or label == "ai" or label.startswith("ai "):
            return score
    for label, score in label_scores:
        if "real" in label or "human" in label:
            return 1.0 - score

    return None


def detect_ai_probability(
    detector: Any,
    text: str,
    max_chars: int,
    *,
    ai_labels: set[str],
    human_labels: set[str],
) -> float | None:
    clipped_text = text[:max(1, max_chars)]
    try:
        raw = detector(clipped_text, top_k=None, truncation=True)
    except TypeError:
        raw = detector(clipped_text)

    if isinstance(raw, list) and raw and isinstance(raw[0], list):
        return _extract_ai_probability(
            raw[0], ai_labels=ai_labels, human_labels=human_labels
        )
    return _extract_ai_probability(raw, ai_labels=ai_labels, human_labels=human_labels)


model_path = ensure_local_model(LOCAL_MODEL_PATH, MODEL_REPO_ID)
model, tokenizer = load_model_and_tokenizer(
    model_path, fix_mistral_regex=FIX_MISTRAL_REGEX
)
detectors = (
    load_ai_detectors(AI_DETECTOR_MODEL_IDS, AI_DETECTOR_CACHE_DIR, token=HF_TOKEN)
    if ENABLE_AI_DETECTOR
    else []
)
prompts, prompt_source = resolve_prompts()

special_token_ids = list(getattr(tokenizer, "all_special_ids", []) or [])

methods = [
    {
        "name": "GREEDY",
        "sampler": make_sampler(temp=0.0),
        "logits_processors": None,
        "seed": None,
    },
    {
        "name": "TOP_P",
        "sampler": make_sampler(temp=0.7, top_p=0.9),
        "logits_processors": None,
        "seed": SAMPLE_SEED,
    },
    {
        "name": "TOP_K",
        "sampler": make_sampler(temp=0.8, top_k=40),
        "logits_processors": None,
        "seed": SAMPLE_SEED,
    },
    {
        "name": "MIN_P",
        "sampler": make_sampler(temp=0.8, min_p=0.08, min_tokens_to_keep=5),
        "logits_processors": None,
        "seed": SAMPLE_SEED,
    },
    {
        "name": "TOP_K_TOP_P",
        "sampler": make_sampler(temp=0.75, top_k=50, top_p=0.92),
        "logits_processors": None,
        "seed": SAMPLE_SEED,
    },
    {
        "name": "REPETITION_PENALTY_TOP_P",
        "sampler": make_sampler(temp=0.7, top_p=0.9),
        "logits_processors": make_logits_processors(
            repetition_penalty=1.1,
            repetition_context_size=64,
        ),
        "seed": SAMPLE_SEED,
    },
    {
        "name": "XTC",
        "sampler": make_sampler(
            temp=0.8,
            top_p=0.95,
            xtc_probability=0.25,
            xtc_threshold=0.1,
            xtc_special_tokens=special_token_ids,
        ),
        "logits_processors": None,
        "seed": SAMPLE_SEED,
    },
    {
        "name": "BEAM_SEARCH",
        "mode": "beam",
        "beam_width": BEAM_WIDTH,
        "length_penalty": BEAM_LENGTH_PENALTY,
        "seed": None,
    },
]

print(f"Using local model: {model_path}")
print(f"Max tokens per method: {MAX_TOKENS}")
print(f"Shared sampling seed: {SAMPLE_SEED}")
print(f"Beam width: {BEAM_WIDTH}, length penalty: {BEAM_LENGTH_PENALTY}")
print(f"HF auth token configured: {'yes' if HF_TOKEN else 'no'}")
print(f"Prompts configured: {len(prompts)}")
print(f"Prompt source: {prompt_source}")
if prompt_source.startswith("MT-Bench"):
    print(f"MT-Bench prompt file: {(MT_BENCH_DATASET_DIR / 'raw' / 'question.jsonl').resolve()}")
    print(f"MT-Bench balanced categories: {'yes' if MT_BENCH_BALANCE_CATEGORIES else 'no'}")
print(f"Sampling runs per prompt: {SAMPLING_RUNS_PER_PROMPT}")
print(f"Tokenizer fix_mistral_regex enabled: {'yes' if FIX_MISTRAL_REGEX else 'no'}")
print(f"Progress bar backend: {'tqdm' if tqdm is not None else 'plain-text fallback'}")
print(f"Print generated text: {'yes' if PRINT_GENERATED_TEXT else 'no'}")
if not ENABLE_AI_DETECTOR:
    print("AI detector disabled (set ENABLE_AI_DETECTOR=1 to enable).")
elif not detectors:
    print(
        "AI detector disabled: install `transformers` (+ backend like `torch`) and verify AI_DETECTOR_MODEL_IDS."
    )
else:
    print(f"AI detector cache dir: {AI_DETECTOR_CACHE_DIR.resolve()}")
    print("AI detector models loaded:")
    for item in detectors:
        print(f"- {item['name']} ({item['label_hint']})")

detector_results: dict[str, dict[str, list[float]]] = defaultdict(
    lambda: defaultdict(list)
)
human_baseline_results: dict[str, list[float]] = defaultdict(list)

prompt_payloads: list[dict[str, Any]] = []
for prompt_idx, prompt_text in enumerate(prompts, start=1):
    messages = [{"role": "user", "content": prompt_text}]
    prompt_tokens = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    prompt_payloads.append(
        {
            "prompt_idx": prompt_idx,
            "prompt_text": prompt_text,
            "prompt_tokens": prompt_tokens,
        }
    )

run_jobs: list[dict[str, Any]] = []
for payload in prompt_payloads:
    for method in methods:
        runs_for_method = SAMPLING_RUNS_PER_PROMPT if method["seed"] is not None else 1
        for run_idx in range(runs_for_method):
            run_jobs.append(
                {
                    "prompt_idx": payload["prompt_idx"],
                    "prompt_text": payload["prompt_text"],
                    "prompt_tokens": payload["prompt_tokens"],
                    "method": method,
                    "run_idx": run_idx,
                    "runs_for_method": runs_for_method,
                }
            )

if detectors and ENABLE_HUMAN_BASELINE and HUMAN_BASELINE_TEXTS:
    baseline_jobs = [
        {"detector_item": detector_item, "text": text}
        for detector_item in detectors
        for text in HUMAN_BASELINE_TEXTS
    ]
    print(f"\nScoring human baselines: {len(baseline_jobs)} runs")
    for baseline_job in iter_with_progress(
        baseline_jobs, desc="Human baseline", unit="runs"
    ):
        detector_item = baseline_job["detector_item"]
        detector_name = detector_item["name"]
        baseline_score = detect_ai_probability(
            detector_item["detector"],
            baseline_job["text"],
            AI_DETECTOR_MAX_CHARS,
            ai_labels=detector_item["ai_labels"],
            human_labels=detector_item["human_labels"],
        )
        if baseline_score is not None:
            human_baseline_results[detector_name].append(baseline_score)

print(f"\nGeneration runs planned: {len(run_jobs)}")
current_prompt_idx = 0
for run_job in iter_with_progress(run_jobs, desc="Generation", unit="runs"):
    prompt_idx = run_job["prompt_idx"]
    prompt_text = run_job["prompt_text"]
    prompt_tokens = run_job["prompt_tokens"]
    method = run_job["method"]
    run_idx = run_job["run_idx"]
    runs_for_method = run_job["runs_for_method"]

    if prompt_idx != current_prompt_idx:
        current_prompt_idx = prompt_idx
        print(f"\nPrompt {prompt_idx}/{len(prompt_payloads)}: {prompt_text}")

    if method["seed"] is not None:
        run_seed = SAMPLE_SEED + ((prompt_idx - 1) * PROMPT_SEED_STRIDE) + run_idx
        mx.random.seed(run_seed)

    if method.get("mode") == "beam":
        output = beam_search_generate(
            model,
            tokenizer,
            prompt_tokens,
            max_tokens=MAX_TOKENS,
            beam_width=method["beam_width"],
            length_penalty=method["length_penalty"],
        )
    else:
        output = generate(
            model,
            tokenizer,
            prompt_tokens,
            sampler=method["sampler"],
            logits_processors=method["logits_processors"],
            max_tokens=MAX_TOKENS,
        )

    run_label = (
        f"{method['name']} [run {run_idx + 1}/{runs_for_method}]"
        if runs_for_method > 1
        else method["name"]
    )
    if PRINT_GENERATED_TEXT:
        print(f"\n{run_label}:\n{output}")
    else:
        print(f"\n{run_label}: complete")

    for detector_item in detectors:
        detector_name = detector_item["name"]
        ai_prob = detect_ai_probability(
            detector_item["detector"],
            output,
            AI_DETECTOR_MAX_CHARS,
            ai_labels=detector_item["ai_labels"],
            human_labels=detector_item["human_labels"],
        )
        if ai_prob is None:
            print(f"Detector {detector_name}: score unavailable (ambiguous labels)")
        else:
            detector_results[detector_name][method["name"]].append(ai_prob)
            print(f"Detector {detector_name} AI-probability: {ai_prob:.3f}")

if detector_results:
    print("\nDetector Summary (higher = more AI-like):")
    for detector_item in detectors:
        detector_name = detector_item["name"]
        detector_scores_by_method = detector_results.get(detector_name)
        if not detector_scores_by_method:
            continue
        print(f"\n{detector_name}:")
        baseline_scores = human_baseline_results.get(detector_name, [])
        if baseline_scores:
            baseline_mean = statistics.fmean(baseline_scores)
            baseline_std = (
                statistics.pstdev(baseline_scores) if len(baseline_scores) > 1 else 0.0
            )
            print(
                f"Human baseline: mean={baseline_mean:.3f}, std={baseline_std:.3f}, n={len(baseline_scores)}"
            )
        else:
            baseline_mean = None

        for method in methods:
            method_name = method["name"]
            scores = detector_scores_by_method.get(method_name, [])
            if not scores:
                continue
            mean_score = statistics.fmean(scores)
            std_score = statistics.pstdev(scores) if len(scores) > 1 else 0.0
            line = (
                f"{method_name}: mean={mean_score:.3f}, std={std_score:.3f}, n={len(scores)}"
            )
            if baseline_mean is not None:
                line += f", delta_vs_human={mean_score - baseline_mean:+.3f}"
            print(line)
