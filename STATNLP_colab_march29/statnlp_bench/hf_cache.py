# Local-first Hugging Face caching layer.

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from .config import env_bool, hf_token_from_env


@dataclass(slots=True)
class HFCacheSettings:
    root: Path
    hub_cache: Path
    datasets_cache: Path
    transformers_cache: Path
    offline: bool
    token: str | None


def _default_cache_root() -> Path:
    return Path(__file__).resolve().parent.parent / "models" / "hf_cache"


# Singleton — resolved once and cached for the process lifetime.
@lru_cache(maxsize=1)
def get_hf_cache_settings() -> HFCacheSettings:
    root = Path(
        os.environ.get("HF_CACHE_ROOT", str(_default_cache_root()))
    ).expanduser().resolve()
    hub_cache = Path(
        os.environ.get("HF_HUB_CACHE")
        or os.environ.get("HUGGINGFACE_HUB_CACHE")
        or root / "hub"
    ).expanduser().resolve()
    datasets_cache = Path(
        os.environ.get("HF_DATASETS_CACHE") or root / "datasets"
    ).expanduser().resolve()
    transformers_cache = Path(
        os.environ.get("TRANSFORMERS_CACHE") or root / "transformers"
    ).expanduser().resolve()
    # Any of these env vars being truthy puts us in offline mode
    offline_vars = ("HF_OFFLINE", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE")
    return HFCacheSettings(
        root=root,
        hub_cache=hub_cache,
        datasets_cache=datasets_cache,
        transformers_cache=transformers_cache,
        offline=any(env_bool(name, False) for name in offline_vars),
        token=hf_token_from_env(),
    )


# Create cache dirs and inject env vars so downstream HF libraries use our paths.
def configure_hf_environment() -> HFCacheSettings:
    settings = get_hf_cache_settings()
    for directory in (settings.root, settings.hub_cache, settings.datasets_cache, settings.transformers_cache):
        directory.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(settings.root))
    os.environ.setdefault("HF_HUB_CACHE", str(settings.hub_cache))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(settings.hub_cache))
    os.environ.setdefault("HF_DATASETS_CACHE", str(settings.datasets_cache))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(settings.transformers_cache))
    if settings.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"
    return settings


def _merge_token(kwargs: dict[str, Any], token: str | None) -> dict[str, Any]:
    if token is None:
        return dict(kwargs)
    merged = dict(kwargs)
    merged.setdefault("token", token)
    return merged


# Wrapper around from_pretrained that handles API differences across
# transformers versions
def _call_from_pretrained(
    loader: Any, model_name_or_path: str | Path, *, token: str | None, **kwargs: Any,
) -> Any:
    try:
        return loader.from_pretrained(model_name_or_path, **_merge_token(kwargs, token))
    except TypeError as exc:
        message = str(exc)
        # Older tokenizers don't accept fix_mistral_regex.
        if "fix_mistral_regex" in kwargs and "fix_mistral_regex" in message:
            cleaned = {k: v for k, v in kwargs.items() if k != "fix_mistral_regex"}
            return loader.from_pretrained(model_name_or_path, **_merge_token(cleaned, token))
        # Older library versions use use_auth_token instead of token.
        legacy = dict(kwargs)
        if token is not None:
            legacy.setdefault("use_auth_token", token)
            legacy.pop("token", None)
        return loader.from_pretrained(model_name_or_path, **legacy)


# Primary entry point for loading models/tokenizers: try cache first, then network.
def from_pretrained_local_first(
    loader: Any,
    model_name_or_path: str | Path,
    *,
    cache_dir: str | Path | None = None,
    token: str | None = None,
    **kwargs: Any,
) -> Any:
    settings = configure_hf_environment()
    resolved_token = token if token is not None else settings.token
    resolved_cache_dir = Path(cache_dir or settings.transformers_cache).expanduser().resolve()
    resolved_cache_dir.mkdir(parents=True, exist_ok=True)

    common = dict(kwargs, cache_dir=str(resolved_cache_dir))
    try:
        return _call_from_pretrained(
            loader, model_name_or_path, token=resolved_token, local_files_only=True, **common,
        )
    except Exception as local_error:
        if settings.offline:
            raise RuntimeError(
                f"Offline mode is enabled and {model_name_or_path!r} is not fully available in {resolved_cache_dir}."
            ) from local_error
    return _call_from_pretrained(loader, model_name_or_path, token=resolved_token, **common)


# Same local-first strategy but for full repo snapshots (e.g. model weights).
def snapshot_download_local_first(
    *,
    repo_id: str,
    local_dir: str | Path,
    allow_patterns: list[str] | None = None,
    token: str | None = None,
) -> str:
    from huggingface_hub import snapshot_download

    settings = configure_hf_environment()
    resolved_token = token if token is not None else settings.token
    local_path = Path(local_dir).expanduser().resolve()
    kwargs: dict[str, Any] = {
        "repo_id": repo_id,
        "local_dir": str(local_path),
        "cache_dir": str(settings.hub_cache),
    }
    if allow_patterns is not None:
        kwargs["allow_patterns"] = allow_patterns
    kwargs = _merge_token(kwargs, resolved_token)
    try:
        return snapshot_download(local_files_only=True, **kwargs)
    except Exception as local_error:
        if settings.offline:
            raise RuntimeError(
                f"Offline mode is enabled and {repo_id!r} is not available in the local Hugging Face cache."
            ) from local_error
    return snapshot_download(**kwargs)


# Same local-first strategy for HF datasets.
def load_dataset_local_first(
    repo_id: str,
    config_name: str | None = None,
    *,
    revision: str | None = None,
    token: str | None = None,
    **kwargs: Any,
) -> Any:
    from datasets import DownloadConfig, load_dataset

    settings = configure_hf_environment()
    resolved_token = token if token is not None else settings.token
    common: dict[str, Any] = dict(kwargs, cache_dir=str(settings.datasets_cache))
    if revision is not None:
        common["revision"] = revision
    if resolved_token is not None:
        common["token"] = resolved_token

    try:
        local = dict(common, download_config=DownloadConfig(
            cache_dir=str(settings.datasets_cache), local_files_only=True,
        ))
        return load_dataset(repo_id, config_name, **local)
    except Exception as local_error:
        if settings.offline:
            raise RuntimeError(
                f"Offline mode is enabled and dataset {repo_id!r} ({config_name!r}) is not available in the local cache."
            ) from local_error

    remote = dict(common, download_config=DownloadConfig(
        cache_dir=str(settings.datasets_cache),
    ))
    return load_dataset(repo_id, config_name, **remote)
