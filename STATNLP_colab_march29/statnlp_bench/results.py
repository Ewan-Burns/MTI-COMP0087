"""Result helpers: IO, plotting, and table rendering."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from .types import MatrixCellResult

# IO helpers

def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def read_json(path: str | Path) -> object:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, payload: object) -> None:
    out = Path(path)
    _ensure_parent(out)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def read_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    out = Path(path)
    _ensure_parent(out)
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: str | Path, rows: Iterable[dict], fieldnames: list[str] | None = None) -> None:
    out = Path(path)
    _ensure_parent(out)
    rows = list(rows)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)



# Heatmap visualisation


# Render the train-method x test-method matrix as a heatmap image.
# Returns None (instead of raising) when matplotlib is not installed,
# so plotting is always optional.
def save_heatmap(
    results: list[MatrixCellResult],
    output_path: str | Path,
    *,
    title: str,
    value_field: str = "accuracy",
) -> Path | None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    row_names = sorted({r.train_method for r in results})
    col_names = sorted({r.test_method for r in results})
    lookup = {(r.train_method, r.test_method): r for r in results}
    matrix = [
        [float(getattr(lookup.get((rn, cn)), value_field, 0) or 0) for cn in col_names]
        for rn in row_names
    ]

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    # Scale figure dimensions to the matrix so labels don't overlap
    fig, ax = plt.subplots(
        figsize=(max(6, len(col_names) * 0.7), max(4, len(row_names) * 0.4)),
    )
    image = ax.imshow(matrix, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(col_names)))
    ax.set_xticklabels(col_names, rotation=45, ha="right")
    ax.set_yticks(range(len(row_names)))
    ax.set_yticklabels(row_names)
    ax.set_title(title)
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)
    return output



# Table rendering

# Flatten MatrixCellResults into plain dicts suitable for CSV/JSONL export.
def matrix_rows(results: list[MatrixCellResult], value_field: str = "accuracy") -> list[dict[str, object]]:
    return [
        {
            "train_method": r.train_method,
            "test_method": r.test_method,
            "detector_name": r.detector_name,
            value_field: getattr(r, value_field),
        }
        for r in results
    ]


# Render the train x test matrix as a tab-separated text table for terminal output.
def render_matrix(results: list[MatrixCellResult], value_field: str = "accuracy") -> str:
    col_names = sorted({r.test_method for r in results})
    row_names = sorted({r.train_method for r in results})
    lookup: dict[tuple[str, str], str] = {}
    for r in results:
        value = getattr(r, value_field)
        lookup[(r.train_method, r.test_method)] = "NA" if value is None else f"{value:.3f}"

    header = ["train\\test"] + col_names
    lines = ["\t".join(header)]
    for rn in row_names:
        lines.append("\t".join([rn] + [lookup.get((rn, cn), "NA") for cn in col_names]))
    return "\n".join(lines)


# Produce one text matrix per detector, keyed by detector name.
def render_matrices_by_detector(
    results: list[MatrixCellResult],
    *,
    value_field: str = "accuracy",
) -> dict[str, str]:
    grouped: dict[str, list[MatrixCellResult]] = defaultdict(list)
    for r in results:
        grouped[r.detector_name].append(r)
    return {
        name: render_matrix(group, value_field=value_field)
        for name, group in sorted(grouped.items())
    }
