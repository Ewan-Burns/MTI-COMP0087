# Preflight checks that verify required ML libraries are importable before
# running a track. Each check spawns a subprocess so an import crash (e.g. bad
# native lib) doesn't kill the main process. Failure messages include concrete
# fix instructions so users don't have to guess at missing dependencies.

from __future__ import annotations

import platform
import subprocess
import sys
from dataclasses import dataclass


@dataclass(slots=True)
class RuntimeCheckResult:
    name: str
    ok: bool
    details: str


class RuntimeCheckError(RuntimeError):
    pass


# Each check is a Python one-liner run in a subprocess; success = exit code 0.
# mbrs sets MPLCONFIGDIR to avoid matplotlib config issues in restricted envs.
_CHECKS = {
    "torch": "import torch; print(torch.__version__)",
    "transformers": "import transformers; print(transformers.__version__)",
    "mbrs": (
        "import os; os.environ.setdefault('MPLCONFIGDIR', '/tmp/statnlp_mpl'); "
        "from mbrs.decoders.mbr import DecoderMBR; "
        "from mbrs.metrics.bertscore import MetricBERTScore; "
        "print('mbrs ok')"
    ),
}


def _run_check(name: str) -> RuntimeCheckResult:
    process = subprocess.run(
        [sys.executable, "-c", _CHECKS[name]],
        capture_output=True,
        text=True,
    )
    if process.returncode == 0:
        details = (process.stdout or process.stderr).strip() or "ok"
        return RuntimeCheckResult(name=name, ok=True, details=details)
    details = (process.stderr or process.stdout).strip() or f"process exited with code {process.returncode}"
    return RuntimeCheckResult(name=name, ok=False, details=details)


def _ensure(track_name: str, check_names: list[str]) -> None:
    results = [_run_check(name) for name in check_names]
    failures = [r for r in results if not r.ok]
    if not failures:
        return
    lines = [
        f"{track_name} runtime preflight failed for interpreter {sys.executable} ({platform.python_version()}).",
        "This repo currently expects a working Python 3.12 environment with PyTorch and Transformers.",
        "",
        "Failed checks:",
        *(f"- {r.name}: {r.details}" for r in failures),
        "",
        "Recommended fix:",
        "- Prefer the pinned environment file in environment.yml.",
        "- Create or activate a Python 3.12 environment.",
        "- Install the benchmark runtime dependencies into that environment.",
        "",
        "Example:",
        "- conda env create -f environment.yml",
        "- conda activate statnlp-bench",
        "- python scripts/check_runtime.py --track all",
    ]
    raise RuntimeCheckError("\n".join(lines))


# ---- Per-track preflight gates ----
# Each function checks exactly the libraries needed for that track.

def ensure_generation_runtime() -> None:
    _ensure("Generative-detection", ["torch", "transformers"])


def ensure_detection_runtime() -> None:
    _ensure("Generative-detection", ["torch", "transformers"])


def ensure_task_runtime() -> None:
    _ensure("Task-efficiency", ["torch", "transformers"])


def ensure_publication_detection_runtime(*, require_mbr: bool = False) -> None:
    checks = ["torch", "transformers"]
    if require_mbr:
        checks.append("mbrs")
    _ensure("Publication detection", checks)
