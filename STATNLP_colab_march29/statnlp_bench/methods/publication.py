
# Implements the full set of sampling strategies from Dubois
# plus novel methods (contrastive search, CFG, Top-H, MBR).

from __future__ import annotations

import hashlib
import importlib.metadata
import os
import time
from functools import lru_cache
from pathlib import Path
from typing import Any
import torch

from ..config import QuickRunConfig
from ..devices import resolve_torch_device
from ..hf_cache import from_pretrained_local_first, snapshot_download_local_first
from ..registry import register_method
from ..types import MethodRunResult, MethodSpec
from .profiles import DUBOIS_FULL, NOVEL_CORE, publication_lane_for_method

# Shared base: ancestral sampling at temp=1.0. All Dubois methods spread-merge
# from this dict, overriding only the parameter under test.

MIN_CONTINUATION_TOKENS = 50

# top_k=0 disables top-k filtering (HF default is 50, which would silently truncate)
_BASE_SAMPLE = {"mode": "sample", "do_sample": True, "temperature": 1.0, "top_k": 0}

TRANSFORMERS_METHODS: dict[str, dict[str, Any]] = {
    "ANCESTRAL":            {**_BASE_SAMPLE},
    "TEMP_05":              {**_BASE_SAMPLE, "temperature": 0.5},
    "TEMP_07":              {**_BASE_SAMPLE, "temperature": 0.7},
    "TEMP_09":              {**_BASE_SAMPLE, "temperature": 0.9},
    "TEMP_11":              {**_BASE_SAMPLE, "temperature": 1.1},
    "TEMP_12":              {**_BASE_SAMPLE, "temperature": 1.2},
    "TEMP_13":              {**_BASE_SAMPLE, "temperature": 1.3},
    "REP_105":              {**_BASE_SAMPLE, "repetition_penalty": 1.05},
    "REP_110":              {**_BASE_SAMPLE, "repetition_penalty": 1.1},
    "REP_115":              {**_BASE_SAMPLE, "repetition_penalty": 1.15},
    "REP_120":              {**_BASE_SAMPLE, "repetition_penalty": 1.2},
    "REP_125":              {**_BASE_SAMPLE, "repetition_penalty": 1.25},
    "REP_130":              {**_BASE_SAMPLE, "repetition_penalty": 1.3},
    "TOP_K_10":             {**_BASE_SAMPLE, "top_k": 10},
    "TOP_K_20":             {**_BASE_SAMPLE, "top_k": 20},
    "TOP_K_50":             {**_BASE_SAMPLE, "top_k": 50},
    "TOP_K_75":             {**_BASE_SAMPLE, "top_k": 75},
    "TOP_K_100":            {**_BASE_SAMPLE, "top_k": 100},
    "TOP_K_1000":           {**_BASE_SAMPLE, "top_k": 1000},
    "TOP_P_03":             {**_BASE_SAMPLE, "top_p": 0.3},
    "TOP_P_05":             {**_BASE_SAMPLE, "top_p": 0.5},
    "TOP_P_07":             {**_BASE_SAMPLE, "top_p": 0.7},
    "TOP_P_08":             {**_BASE_SAMPLE, "top_p": 0.8},
    "TOP_P_09":             {**_BASE_SAMPLE, "top_p": 0.9},
    "TOP_P_095":            {**_BASE_SAMPLE, "top_p": 0.95},
    "TYPICAL_03":           {**_BASE_SAMPLE, "typical_p": 0.3},
    "TYPICAL_05":           {**_BASE_SAMPLE, "typical_p": 0.5},
    "TYPICAL_07":           {**_BASE_SAMPLE, "typical_p": 0.7},
    "TYPICAL_08":           {**_BASE_SAMPLE, "typical_p": 0.8},
    "TYPICAL_09":           {**_BASE_SAMPLE, "typical_p": 0.9},
    "TYPICAL_095":          {**_BASE_SAMPLE, "typical_p": 0.95},
    "ETA_1E4":              {**_BASE_SAMPLE, "eta_cutoff": 1e-4},
    "ETA_5E3":              {**_BASE_SAMPLE, "eta_cutoff": 5e-3},
    "ETA_1E3":              {**_BASE_SAMPLE, "eta_cutoff": 1e-3},
    "ETA_01":               {**_BASE_SAMPLE, "eta_cutoff": 1e-2},
    "ETA_05":               {**_BASE_SAMPLE, "eta_cutoff": 5e-2},
    "ETA_10":               {**_BASE_SAMPLE, "eta_cutoff": 1e-1},
    "CONTRASTIVE_K8_A06":   {"mode": "contrastive", "top_k": 8, "penalty_alpha": 0.6},
    "CFG_20":               {"mode": "cfg", "do_sample": True, "guidance_scale": 2.0, "temperature": 1.0},
}

# Minimum Bayes Risk decoding: generate N candidates, pick the one with
# highest expected utility (BERTScore F1) against all others.
MBR_METHODS: dict[str, dict[str, Any]] = {
    "MBR_16_BERTSCORE": {
        "mode": "mbr",
        "num_candidates": 16,
        "proposal_kwargs": {"do_sample": True, "temperature": 0.9, "top_p": 0.95},
        "metric_name": "bertscore_f1",
        "metric_model": "roberta-large",
    },
}

# Provenance tracking: links each method family to its paper/implementation
SOURCE_METADATA: dict[str, dict[str, str]] = {
    "dubois_sampling": {
        "source_url": "https://aclanthology.org/2025.findings-emnlp.609.pdf",
        "source_version": "dubois-2025",
        "implementation": "transformers.generate",
    },
    "contrastive": {
        "source_url": "https://arxiv.org/abs/2202.06417",
        "source_version": "transformers.generate",
        "implementation": "transformers contrastive search",
    },
    "cfg": {
        "source_url": "https://arxiv.org/abs/2306.17806",
        "source_version": "transformers.generate",
        "implementation": "transformers classifier-free guidance",
    },
    "p_less": {
        "source_url": "https://openreview.net/forum?id=21455",
        "source_version": "custom logits processor",
        "implementation": "transformers P-less (Tan et al., ICLR 2026)",
    },
    "top_h": {
        "source_url": "https://arxiv.org/abs/2509.02510",
        "source_version": "custom logits processor",
        "implementation": "transformers Top-H",
    },
    "mbr": {
        "source_url": "https://github.com/naist-nlp/mbrs",
        "source_version": "official mbrs library",
        "implementation": "mbrs DecoderMBR + MetricBERTScore",
    },
}

CUSTOM_GENERATE_REPOS: dict[str, str] = {
    "contrastive": "transformers-community/contrastive-search",
}

# Maps method-name prefixes to SOURCE_METADATA keys for provenance lookup
_SOURCE_PREFIX_MAP = {
    "CONTRASTIVE_": "contrastive",
    "CFG_": "cfg",
    "P_LESS": "p_less",
    "TOP_H_": "top_h",
    "MBR_": "mbr",
}

_TORCH_DTYPE_MAP = {
    "float32": "float32", "fp32": "float32",
    "float16": "float16", "fp16": "float16",
    "bfloat16": "bfloat16", "bf16": "bfloat16",
}



# Custom logits processors for novel sampling methods


def _make_p_less_processor():
    """P-less sampling: mask tokens with probability below sum_i(p_i^2).

    The threshold adapts to the shape of the distribution — peaky
    distributions get a higher threshold (more aggressive masking),
    flat distributions get a lower one.
    """
    

    class PLessProcessor:
        def __call__(self, input_ids, scores):
            probs = torch.softmax(scores, dim=-1)
            threshold = (probs * probs).sum(dim=-1, keepdim=True)
            scores[probs < threshold] = -float("inf")
            return scores

    return PLessProcessor()


def _make_top_h_processor(alpha: float):
    """Top-H sampling: greedily add tokens until the renormalized subset
    entropy H(q) would exceed alpha * H(p), where H(p) is the full
    distribution entropy.

    Lower alpha = stricter (fewer tokens kept), higher alpha = more diverse.
    """

    class TopHProcessor:
        def __call__(self, input_ids, scores):
            probs = torch.softmax(scores, dim=-1)
            # H(p) — entropy of the full distribution
            log_probs = torch.log(probs.clamp(min=1e-12))
            Hp = -(probs * log_probs).sum(dim=-1)  # (batch,)
            limit = alpha * Hp

            sorted_probs, sorted_indices = probs.sort(dim=-1, descending=True)

            # Cumulative quantities for running entropy computation
            cumsum = sorted_probs.cumsum(dim=-1)
            cum_plogp = (sorted_probs * sorted_probs.clamp(min=1e-12).log()).cumsum(dim=-1)
            Hq = cumsum.clamp(min=1e-12).log() - cum_plogp / cumsum.clamp(min=1e-12)

            # Keep tokens where H(q) <= limit; always keep at least 1
            keep_mask = Hq <= limit.unsqueeze(-1)
            keep_mask[..., 0] = True  # always keep the top token

            # Map back to original vocab ordering
            original_mask = torch.zeros_like(scores, dtype=torch.bool)
            original_mask.scatter_(dim=-1, index=sorted_indices, src=keep_mask)
            scores[~original_mask] = -float("inf")
            return scores

    return TopHProcessor()


# Maps method names to (processor_factory, base_generation_kwargs)
_CUSTOM_PROCESSOR_METHODS: dict[str, tuple[Any, dict[str, Any]]] = {
    "P_LESS": (_make_p_less_processor, {**_BASE_SAMPLE, "temperature": 0.8}),
    "TOP_H_05": (_make_top_h_processor, {**_BASE_SAMPLE, "temperature": 0.8}),
    "TOP_H_07": (_make_top_h_processor, {**_BASE_SAMPLE, "temperature": 0.8}),
}

# Alpha values for Top-H variants
_TOP_H_ALPHAS: dict[str, float] = {
    "TOP_H_05": 0.5,
    "TOP_H_07": 0.7,
}


def supports_publication_dataset(dataset_name: str) -> bool:
    normalized = (dataset_name or "").strip().lower()
    return normalized not in {"", "mt_bench_local"}


# Lazy import: keeps torch/transformers out of the module-level import graph
# so the benchmark can discover methods without loading heavy libraries.
def _lazy_import_torch():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
    return torch, AutoTokenizer, AutoModelForCausalLM, GenerationConfig


def _torch_dtype(dtype_name: str, torch_module):
    canonical = _TORCH_DTYPE_MAP.get((dtype_name or "auto").strip().lower())
    return getattr(torch_module, canonical) if canonical else None


# matplotlib (transitive dep of mbrs/BERTScore) needs a writable config dir
def _ensure_mpl_config_dir() -> None:
    if os.environ.get("MPLCONFIGDIR"):
        return
    cache_dir = Path("./models/hf_cache/matplotlib").expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(cache_dir)


@lru_cache(maxsize=4)
def _load_publication_model_cached(
    model_id: str,
    revision: str | None,
    device: str,
    torch_dtype_name: str,
    attn_implementation: str | None,
) -> tuple[Any, Any]:
    torch, AutoTokenizer, AutoModelForCausalLM, _ = _lazy_import_torch()
    tokenizer = from_pretrained_local_first(
        AutoTokenizer, model_id, revision=revision,
        use_fast=True, trust_remote_code=True,
    )

    model_kwargs: dict[str, Any] = {"revision": revision, "trust_remote_code": True}
    resolved_dtype = _torch_dtype(torch_dtype_name, torch)
    if resolved_dtype is not None:
        model_kwargs["torch_dtype"] = resolved_dtype
    if attn_implementation:
        model_kwargs["attn_implementation"] = attn_implementation

    model = from_pretrained_local_first(AutoModelForCausalLM, model_id, **model_kwargs)

    # Many causal LMs ship without a pad token; reuse EOS to avoid errors
    if getattr(tokenizer, "pad_token_id", None) is None and getattr(tokenizer, "eos_token_id", None) is not None:
        tokenizer.pad_token = tokenizer.eos_token
    model = model.to(device)
    model.eval()
    return tokenizer, model


def _load_model(config: QuickRunConfig) -> tuple[Any, Any, Any, str]:
    """Returns (torch_module, tokenizer, model, device)."""
    torch, _, _, _ = _lazy_import_torch()
    device = resolve_torch_device(config.model.publication_device)
    tokenizer, model = _load_publication_model_cached(
        config.model.publication_model_id,
        config.model.publication_model_revision,
        device,
        config.model.publication_torch_dtype,
        config.model.publication_attn_implementation,
    )
    return torch, tokenizer, model, device


# Seed determinism: unique seed per (prompt, run) 
def _prompt_seed(config: QuickRunConfig, prompt_idx: int, run_idx: int) -> int:
    return int(config.sample_seed + ((prompt_idx - 1) * config.prompt_seed_stride) + run_idx)


def _set_torch_seed(torch_mod, seed: int) -> None:
    torch_mod.manual_seed(seed)
    if torch_mod.cuda.is_available():
        torch_mod.cuda.manual_seed_all(seed)


def _is_instruct_model(tokenizer) -> bool:
    """Check if the model is an instruct/chat model based on its name."""
    name = getattr(tokenizer, "name_or_path", "") or ""
    return any(tag in name.lower() for tag in ("instruct", "chat", "it-"))


def _chat_prompt(tokenizer, prompt_text: str) -> str:
    # Base models should receive raw text for continuation — no chat wrapping.
    # Only instruct/chat models get the chat template applied.
    if not _is_instruct_model(tokenizer):
        return prompt_text
    messages = [{"role": "user", "content": prompt_text}]
    try:
        return tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    except Exception:
        return prompt_text


# "mode" is our routing key, not a transformers kwarg — strip before passing
def _strip_mode(method_kwargs: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in method_kwargs.items() if k != "mode"}


def _make_generation_config(tokenizer, *, max_tokens: int, kwargs: dict[str, Any]):
    _, _, _, GenerationConfig = _lazy_import_torch()
    config = GenerationConfig(
        max_new_tokens=max_tokens,
        pad_token_id=getattr(tokenizer, "pad_token_id", None),
        eos_token_id=getattr(tokenizer, "eos_token_id", None),
    )
    for key, value in kwargs.items():
        setattr(config, key, value)
    return config


@lru_cache(maxsize=8)
def _resolve_custom_generate_path(repo_id: str) -> str:
    local_dir = Path("./models/hf_cache/custom_generate") / repo_id.replace("/", "--")
    try:
        return snapshot_download_local_first(
            repo_id=repo_id, local_dir=local_dir, allow_patterns=["custom_generate/*"],
        )
    except RuntimeError as exc:
        raise RuntimeError(
            f"Official custom generation repo {repo_id!r} is required for this publication method. "
            "Run once online to warm the local cache, then rerun offline if needed."
        ) from exc


def _custom_generate_kwargs(mode: str) -> dict[str, Any]:
    repo_id = CUSTOM_GENERATE_REPOS.get(mode)
    if repo_id is None:
        return {}
    return {"custom_generate": _resolve_custom_generate_path(repo_id)}


def _build_logits_processors(method_name: str) -> list[Any]:
    """Build custom logits processors for methods that need them."""
    if method_name not in _CUSTOM_PROCESSOR_METHODS:
        return []
    factory, _ = _CUSTOM_PROCESSOR_METHODS[method_name]
    if method_name in _TOP_H_ALPHAS:
        return [factory(_TOP_H_ALPHAS[method_name])]
    return [factory()]


def _run_single_prompt(
    prompt_text: str,
    method_name: str,
    method_kwargs: dict[str, Any],
    prompt_idx: int,
    run_idx: int,
    config: QuickRunConfig,
    torch_mod,
    tokenizer,
    model,
    device: str,
) -> tuple[str, dict[str, Any]]:
    '''
    Runs a single prompt according to a sampling method
    '''
    seed = _prompt_seed(config, prompt_idx, run_idx)
    _set_torch_seed(torch_mod, seed)

    prompt = _chat_prompt(tokenizer, prompt_text)
    encoded = {k: v.to(device) for k, v in tokenizer(prompt, truncation=True, return_tensors="pt").items()}
    input_length = int(encoded["input_ids"].shape[-1])
    mode = method_kwargs.get("mode", "sample")

    gen_kwargs = _strip_mode(method_kwargs)
    gen_config = _make_generation_config(tokenizer, max_tokens=config.max_tokens, kwargs=gen_kwargs)

    generate_args: dict[str, Any] = {**encoded, "generation_config": gen_config}
    generate_args.update(_custom_generate_kwargs(mode))

    # Inject custom logits processors for stuff like (P-less, Top-H, etc.)
    processors = _build_logits_processors(method_name)
    if processors:
        from transformers import LogitsProcessorList
        generate_args["logits_processor"] = LogitsProcessorList(processors)

    if mode == "cfg":
        neg_text = config.model.publication_negative_prompt or tokenizer.pad_token or tokenizer.eos_token or " "
        neg = _chat_prompt(tokenizer, neg_text)
        neg_encoded = {k: v.to(device) for k, v in tokenizer(neg, truncation=True, return_tensors="pt").items()}
        generate_args["negative_prompt_ids"] = neg_encoded["input_ids"]
        generate_args["negative_prompt_attention_mask"] = neg_encoded["attention_mask"]

    is_deterministic = not method_kwargs.get("do_sample", False)
    max_retries = 1 if is_deterministic else 10
    for _attempt in range(max_retries):
        with torch_mod.no_grad():
            output_ids = model.generate(**generate_args, trust_remote_code=True)
        generated_tokens = output_ids[0, input_length:]
        if len(generated_tokens) >= MIN_CONTINUATION_TOKENS:
            break

    text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    return text, {"seed": seed, "input_length": input_length}


# Batched generation: left-padding is required for causal LMs so that all
# sequences in the batch share a common right-aligned generation position.
def _run_batched_sample(
    prompts: list[str],
    method_name: str,
    method_kwargs: dict[str, Any],
    prompt_start_idx: int,
    run_idx: int,
    config: QuickRunConfig,
    torch_mod,
    tokenizer,
    model,
    device: str,
) -> tuple[list[str], list[dict[str, Any]]]:
    
    seed = _prompt_seed(config, prompt_start_idx, run_idx)
    _set_torch_seed(torch_mod, seed)

    chat_prompts = [_chat_prompt(tokenizer, p) for p in prompts]

    # Temporarily switch to left-padding, then restore to avoid side-effects
    original_side = getattr(tokenizer, "padding_side", "right")
    tokenizer.padding_side = "left"
    try:
        encoded = tokenizer(chat_prompts, return_tensors="pt", padding=True, truncation=True)
    finally:
        tokenizer.padding_side = original_side

    encoded = {k: v.to(device) for k, v in encoded.items()}
    padded_length = encoded["input_ids"].shape[-1]
    input_lengths = encoded["attention_mask"].sum(dim=-1)

    gen_kwargs = _strip_mode(method_kwargs)
    gen_config = _make_generation_config(tokenizer, max_tokens=config.max_tokens, kwargs=gen_kwargs)

    mode = method_kwargs.get("mode", "sample")
    generate_args: dict[str, Any] = {**encoded, "generation_config": gen_config}
    generate_args.update(_custom_generate_kwargs(mode))
    processors = _build_logits_processors(method_name)
    if processors:
        from transformers import LogitsProcessorList
        generate_args["logits_processor"] = LogitsProcessorList(processors)
    if mode == "cfg":
        batch_n = len(prompts)
        neg_text = config.model.publication_negative_prompt or tokenizer.pad_token or tokenizer.eos_token or " "
        neg = _chat_prompt(tokenizer, neg_text)
        neg_encoded = {k: v.to(device) for k, v in tokenizer(neg, truncation=True, return_tensors="pt").items()}
        # Expand to match batch size — HF expects negative_prompt_ids.shape[0] == batch_size
        generate_args["negative_prompt_ids"] = neg_encoded["input_ids"].expand(batch_n, -1)
        generate_args["negative_prompt_attention_mask"] = neg_encoded["attention_mask"].expand(batch_n, -1)

    with torch_mod.no_grad():
        output_ids = model.generate(**generate_args, trust_remote_code=True)

    texts = []
    per_text = []
    short_indices = []
    # Slice from padded_length onward to skip both padding and input tokens
    for i in range(len(prompts)):
        generated_tokens = output_ids[i, padded_length:]
        texts.append(tokenizer.decode(generated_tokens, skip_special_tokens=True).strip())
        per_text.append({"seed": seed, "batch_seed": True, "input_length": int(input_lengths[i])})
        if len(generated_tokens) < MIN_CONTINUATION_TOKENS:
            short_indices.append(i)

    # Retry short outputs individually (like DUbois)
    for i in short_indices:
        text, metadata = _run_single_prompt(
            prompts[i], method_name, method_kwargs,
            prompt_start_idx + i, run_idx,
            config, torch_mod, tokenizer, model, device,
        )
        texts[i] = text
        per_text[i] = metadata

    return texts, per_text


@lru_cache(maxsize=4)
def _load_mbr_decoder(metric_model: str, device: str) -> tuple[Any, Any]:
    _ensure_mpl_config_dir()
    from mbrs.decoders.mbr import DecoderMBR
    from mbrs.metrics.bertscore import BERTScoreScoreType, MetricBERTScore

    metric = MetricBERTScore(
        MetricBERTScore.Config(
            score_type=BERTScoreScoreType.f1,
            model_type=metric_model,
            cpu=not str(device).startswith("cuda"),
        )
    )
    decoder = DecoderMBR(DecoderMBR.Config(), metric)
    return metric, decoder


def _mbr_select_candidate(
    candidates: list[str], *, metric_model: str, device: str,
) -> tuple[str, dict[str, Any]]:
    """
    Selects candidate for MBR
    """
    if not candidates:
        return "", {"selected_index": 0, "utilities": []}
    metric, decoder = _load_mbr_decoder(metric_model, device)
    expected_scores = metric.expected_scores(candidates, candidates).detach().cpu().tolist()
    output = decoder.decode(candidates, candidates, nbest=1)
    selected = int(output.idx[0]) if output.idx else 0
    return candidates[selected], {
        "selected_index": selected,
        "utilities": [float(s) for s in expected_scores],
        "decoder_scores": [float(s) for s in output.score],
        "num_candidates": len(candidates),
        "mbr_backend": "mbrs",
    }


def _run_mbr_generation(
    prompts: list[str],
    method_name: str,
    prompt_start_idx: int,
    run_idx: int,
    config: QuickRunConfig,
) -> MethodRunResult:
    """
    Runs MBR
    """
    torch_mod, tokenizer, model, device = _load_model(config)
    method_kwargs = MBR_METHODS[method_name]
    num_candidates = method_kwargs["num_candidates"]
    started = time.perf_counter()
    outputs: list[str] = []
    per_text: list[dict[str, Any]] = []

    # Batch generation: N prompts × num_candidates sequences at once.
    # Left-pad so all prompts share a common generation start position.
    seed = _prompt_seed(config, prompt_start_idx, run_idx)
    _set_torch_seed(torch_mod, seed)

    chat_prompts = [_chat_prompt(tokenizer, p) for p in prompts]
    original_side = getattr(tokenizer, "padding_side", "right")
    tokenizer.padding_side = "left"
    try:
        encoded = tokenizer(chat_prompts, return_tensors="pt", padding=True, truncation=True)
    finally:
        tokenizer.padding_side = original_side

    encoded = {k: v.to(device) for k, v in encoded.items()}
    padded_length = encoded["input_ids"].shape[-1]

    gen_config = _make_generation_config(
        tokenizer, max_tokens=config.max_tokens, kwargs=method_kwargs["proposal_kwargs"],
    )
    with torch_mod.no_grad():
        generated = model.generate(
            **encoded, generation_config=gen_config,
            num_return_sequences=num_candidates,
            trust_remote_code=True,
        )

    # generated shape: (len(prompts) * num_candidates, seq_len)
    # Reshape to (len(prompts), num_candidates, seq_len) for per-prompt selection
    for offset in range(len(prompts)):
        start_seq = offset * num_candidates
        end_seq = start_seq + num_candidates
        candidates = [
            tokenizer.decode(generated[i, padded_length:], skip_special_tokens=True).strip()
            for i in range(start_seq, end_seq)
        ]
        chosen, metadata = _mbr_select_candidate(
            candidates, metric_model=method_kwargs["metric_model"], device=device,
        )
        outputs.append(chosen)
        per_text.append({
            **metadata,
            "seed": seed, "batch_seed": True,
            "metric_name": method_kwargs["metric_name"],
            "metric_model": method_kwargs["metric_model"],
        })

    elapsed = (time.perf_counter() - started) * 1000.0
    return MethodRunResult(
        method_name=method_name, texts=outputs,
        latency_ms=elapsed, metadata={"per_text": per_text},
    )


def run_publication_generation_method(
    *,
    prompts: list[str],
    method_name: str,
    config: QuickRunConfig,
    prompt_start_idx: int = 1,
    run_idx: int = 0,
    **_: Any,
) -> MethodRunResult:
    """
    Main method where generation occurrs, loads every type and acts accordingly
    """
    if method_name in MBR_METHODS:
        return _run_mbr_generation(
            prompts, method_name, prompt_start_idx, run_idx, config,
        )

    torch_mod, tokenizer, model, device = _load_model(config)

    # Custom-processor methods store their base kwargs separately
    if method_name in _CUSTOM_PROCESSOR_METHODS:
        _, method_kwargs = _CUSTOM_PROCESSOR_METHODS[method_name]
    else:
        method_kwargs = TRANSFORMERS_METHODS[method_name]
    mode = method_kwargs.get("mode", "sample")
    started = time.perf_counter()

    if len(prompts) > 1:
        outputs, per_text = _run_batched_sample(
            prompts, method_name, method_kwargs, prompt_start_idx, run_idx,
            config, torch_mod, tokenizer, model, device,
        )
    else:
        outputs = []
        per_text = []
        for offset, prompt_text in enumerate(prompts):
            text, metadata = _run_single_prompt(
                prompt_text, method_name, method_kwargs, prompt_start_idx + offset, run_idx,
                config, torch_mod, tokenizer, model, device,
            )
            outputs.append(text)
            per_text.append(metadata)

    elapsed = (time.perf_counter() - started) * 1000.0
    return MethodRunResult(
        method_name=method_name, texts=outputs,
        latency_ms=elapsed, metadata={"per_text": per_text},
    )


def _method_source_metadata(method_name: str) -> dict[str, str]:
    if method_name in DUBOIS_FULL:
        return SOURCE_METADATA["dubois_sampling"]
    for prefix, key in _SOURCE_PREFIX_MAP.items():
        if method_name.startswith(prefix):
            return SOURCE_METADATA[key]
    return {"source_url": "", "source_version": "", "implementation": "unknown"}


# Fingerprint for cache-busting: changes when model/method config changes
def _config_hash(method_name: str, metadata: dict[str, Any], config: QuickRunConfig) -> str:
    payload = repr((
        method_name,
        metadata.get("source_url", ""),
        config.model.publication_model_id,
        config.model.publication_model_revision,
        config.model.publication_torch_dtype,
        config.model.publication_negative_prompt,
        config.model.publication_attn_implementation,
        config.max_tokens,
        config.sample_seed,
        config.prompt_seed_stride,
    ))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _method_metadata(method_name: str) -> dict[str, Any]:
    provenance = _method_source_metadata(method_name)
    metadata: dict[str, Any] = {
        "backend": "publication_torch",
        "publication_fair": True,
        "paper_lane": publication_lane_for_method(method_name),
        "source_url": provenance["source_url"],
        "source_version": provenance["source_version"],
        "source_implementation": provenance["implementation"],
        "third_party_manifest": "third_party/official_methods.json",
    }
    if method_name in NOVEL_CORE:
        metadata["family"] = "novel"
    elif method_name in DUBOIS_FULL:
        metadata["family"] = "dubois"
    return metadata


def available_publication_method_names() -> list[str]:
    return [*DUBOIS_FULL, *NOVEL_CORE]


def publication_metadata_for_method(
    method_name: str, config: QuickRunConfig | None = None,
) -> dict[str, Any]:
    metadata = _method_metadata(method_name)
    if config is not None:
        metadata["model_id"] = config.model.publication_model_id
        metadata["model_revision"] = config.model.publication_model_revision
        metadata["config_hash"] = _config_hash(method_name, metadata, config)
    if method_name.startswith("MBR_"):
        try:
            metadata["source_version"] = importlib.metadata.version("mbrs")
        except importlib.metadata.PackageNotFoundError:
            pass
    return metadata


# Register all publication methods at import time
def _register_publication_methods() -> None:
    for method_name in available_publication_method_names():
        register_method(
            MethodSpec(
                name=method_name,
                track="generative_detection",
                family="publication_generation",
                supports_answer_voting=False,
                supports_dataset=supports_publication_dataset,
                run=run_publication_generation_method,
                metadata=_method_metadata(method_name),
            )
        )


_register_publication_methods()
