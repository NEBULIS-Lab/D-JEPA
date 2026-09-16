from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts" / "robotics"
    / "build_candidates.py"
)


def _module():
    spec = importlib.util.spec_from_file_location("robotwin_grasp_geometry_candidates", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_convert_and_compose_geometry_candidates() -> None:
    raw = np.zeros((6, 20), dtype=np.float32)
    raw[:, 0:3] = (0.0, 0.0, 0.9)
    raw[:, 3] = raw[:, 7] = 1.0
    raw[:, 9] = 1.0
    raw[:, 10:13] = (0.2, 0.0, 0.9)
    raw[:, 13] = raw[:, 17] = 1.0
    raw[:, 19] = 1.0
    raw[3:, 9] = raw[3:, 19] = 0.0
    native = np.zeros((10, 16), dtype=np.float32)
    native[:, 3] = native[:, 11] = 1.0
    hypotheses = np.stack(
        [np.asarray([0.01 * i, 0.0, 0.9, 0.2 + 0.01 * i, 0.0, 0.9]) for i in range(16)]
    )

    converted = _module()._eef20_to_ee16(raw)
    candidates = _module()._compose_candidates(
        native,
        converted,
        source_target=np.asarray([0.0, 0.0, 0.9, 0.2, 0.0, 0.9]),
        target_hypotheses=hypotheses,
        grasp_step=3,
    )

    assert converted.shape == (6, 16)
    assert candidates.shape == (17, 10, 16)
    np.testing.assert_array_equal(candidates[0], native)
    np.testing.assert_allclose(candidates[1, 3, :3], hypotheses[0, :3], atol=1e-6)
    np.testing.assert_allclose(candidates[-1, 3, 8:11], hypotheses[-1, 3:], atol=1e-6)
    assert np.isfinite(candidates).all()


def test_compose_extends_early_success_native_to_demo_horizon() -> None:
    native = np.zeros((3, 16), dtype=np.float32)
    demo = np.zeros((7, 16), dtype=np.float32)
    native[:, 3] = native[:, 11] = 1.0
    demo[:, 3] = demo[:, 11] = 1.0
    source = np.asarray([0.0, 0.0, 0.9, 0.2, 0.0, 0.9])
    demo[5, :3] = source[:3]
    demo[5, 8:11] = source[3:]
    hypotheses = np.repeat(source[None], 16, axis=0)

    candidates = _module()._compose_candidates(
        native,
        demo,
        source_target=source,
        target_hypotheses=hypotheses,
        grasp_step=5,
    )

    assert candidates.shape == (17, 7, 16)
    np.testing.assert_array_equal(candidates[0, :3], native)
    np.testing.assert_array_equal(candidates[0, 3:], np.repeat(native[-1:], 4, axis=0))
