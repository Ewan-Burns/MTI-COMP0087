

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterable, Iterator

from .config import env_bool

try:
    from tqdm.auto import tqdm as _tqdm
except ImportError:  # pragma: no cover - fallback path
    _tqdm = None


def progress_enabled() -> bool:
    return env_bool("STATNLP_PROGRESS", True)


# No-op stand-in for a tqdm bar — used when tqdm is missing or disabled.
@dataclass(slots=True)
class NullProgress:
    total: int | None = None
    desc: str = ""

    def update(self, _: int = 1) -> None:
        pass

    def set_description_str(self, _: str) -> None:
        pass

    def set_postfix(self, ordered_dict: dict[str, Any] | None = None, refresh: bool = True, **kwargs: Any) -> None:
        pass

    def set_postfix_str(self, _: str, refresh: bool = True) -> None:
        pass

    def write(self, message: str) -> None:
        print(message)

    def close(self) -> None:
        pass


def progress_write(message: str) -> None:
    if _tqdm is not None and progress_enabled():
        _tqdm.write(message)
    else:
        print(message)


# Wrap an iterable with a progress bar (like tqdm(iterable, ...)).
def progress_iter(
    iterable: Iterable[Any],
    *,
    desc: str,
    total: int | None = None,
    unit: str = "item",
    leave: bool = False,
    disable: bool | None = None,
) -> Iterable[Any]:
    if _tqdm is None:
        return iterable
    resolved_disable = (not progress_enabled()) if disable is None else disable
    return _tqdm(
        iterable,
        total=total,
        desc=desc,
        unit=unit,
        leave=leave,
        dynamic_ncols=True,
        disable=resolved_disable,
    )


# Context-manager variant for manual .update() calls inside a loop.
@contextmanager
def progress_task(
    *,
    total: int | None,
    desc: str,
    unit: str = "item",
    leave: bool = True,
    disable: bool | None = None,
) -> Iterator[Any]:
    if _tqdm is None:
        yield NullProgress(total=total, desc=desc)
        return
    resolved_disable = (not progress_enabled()) if disable is None else disable
    with _tqdm(
        total=total,
        desc=desc,
        unit=unit,
        leave=leave,
        dynamic_ncols=True,
        disable=resolved_disable,
    ) as progress:
        yield progress
