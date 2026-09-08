"""Safe, non-overwriting I/O helpers for native reproduction artifacts."""

from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
import secrets
import sys
from typing import Any, Mapping, Sequence

import numpy as np

from .config import sha256_file


def load_npz(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(Path(path), allow_pickle=False) as archive:
        return {name: np.array(archive[name], copy=True) for name in archive.files}


def _stage_path(final: Path) -> Path:
    final.parent.mkdir(parents=True, exist_ok=True)
    return final.with_name(f".{final.stem}.stage.{os.getpid()}.{secrets.token_hex(8)}{final.suffix}")


def _promote(stage: Path, final: Path) -> None:
    try:
        os.link(stage, final)
    except FileExistsError as error:
        raise FileExistsError(f"refusing to replace existing output: {final}") from error
    finally:
        if stage.exists():
            stage.unlink()


def save_npz(path: str | Path, arrays: Mapping[str, Any]) -> str:
    final = Path(path)
    if final.exists():
        raise FileExistsError(f"refusing to replace existing output: {final}")
    stage = _stage_path(final)
    try:
        with stage.open("xb") as stream:
            np.savez_compressed(stream, **{key: np.asarray(value) for key, value in arrays.items()})
            stream.flush()
            os.fsync(stream.fileno())
        digest = sha256_file(stage)
        _promote(stage, final)
        return digest
    finally:
        if stage.exists():
            stage.unlink()


def save_json(path: str | Path, value: Mapping[str, Any]) -> None:
    final = Path(path)
    if final.exists():
        raise FileExistsError(f"refusing to replace existing output: {final}")
    stage = _stage_path(final)
    try:
        payload = json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
        with stage.open("x", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        _promote(stage, final)
    finally:
        if stage.exists():
            stage.unlink()


def prepend_python_roots(roots: Sequence[str | Path]) -> None:
    """Expose explicitly supplied upstream source checkouts for this process."""
    resolved = [str(Path(root).resolve()) for root in roots]
    for root in resolved:
        if not Path(root).is_dir():
            raise ValueError(f"upstream Python root is not a directory: {root}")
    for root in reversed(resolved):
        if root not in sys.path:
            sys.path.insert(0, root)
    importlib.invalidate_caches()


__all__ = ["load_npz", "prepend_python_roots", "save_json", "save_npz"]
