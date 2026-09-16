from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "robotics" / "candidate_subset.py"


def _module():
    spec = importlib.util.spec_from_file_location("robotwin_candidate_subset", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_subset_preserves_original_ids_and_remaps_native() -> None:
    actions = np.arange(5 * 3 * 16, dtype=np.float32).reshape(5, 3, 16)
    sources = [{"candidate_index": i, "role": "native" if i == 0 else "proposal"} for i in range(5)]

    selected, rows, native = _module()._subset(actions, sources, (0, 3), native_index=0)

    np.testing.assert_array_equal(selected, actions[[0, 3]])
    assert native == 0
    assert rows[1]["candidate_index"] == 1
    assert rows[1]["original_candidate_index"] == 3
    with pytest.raises(ValueError, match="unique"):
        _module()._subset(actions, sources, (0, 0), native_index=0)
