"""Add the repository root to sys.path so scripts can import statnlp_bench."""
from __future__ import annotations

import sys
from pathlib import Path


def bootstrap_repo_path() -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)
    return repo_root
