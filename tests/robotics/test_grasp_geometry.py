from __future__ import annotations

import numpy as np

from djepa.robotics import grasp_geometry
from djepa.robotics.grasp_geometry import (
    colored_tool_geometry,
    fit_geometry_ridge,
    grasp_target,
    normalize_target_span,
    spatial_target_grid,
    warp_ee_trajectory,
)


def test_colored_tool_geometry_is_color_invariant() -> None:
    blue = np.full((240, 320, 3), 245, dtype=np.uint8)
    orange = blue.copy()
    for image, color in ((blue, (40, 100, 210)), (orange, (220, 130, 50))):
        for offset in range(-3, 4):
            x = np.arange(90, 231)
            y = (210 - 0.4 * (x - 90) + offset).astype(int)
            image[y, x] = color

    blue_geometry = colored_tool_geometry(blue)
    orange_geometry = colored_tool_geometry(orange)

    np.testing.assert_allclose(blue_geometry, orange_geometry, atol=1e-6)
    assert blue_geometry.shape == (9,)
    assert blue_geometry[2] > 0.0


def test_geometry_ridge_selects_lambda_only_from_calibration() -> None:
    rng = np.random.default_rng(3)
    features = rng.normal(size=(12, 9))
    targets = np.c_[features[:, :3], features[:, 3:6]]
    model = fit_geometry_ridge(
        features[:8], targets[:8], features[8:], targets[8:], lambdas=(0.0, 0.1, 10.0)
    )

    prediction = model.predict(features[8:])
    assert prediction.shape == (4, 6)
    assert model.lambda_value in {0.0, 0.1, 10.0}
    assert np.isfinite(model.calibration_rmse)


def test_grasp_target_and_span_normalization() -> None:
    actions = np.zeros((10, 20), dtype=np.float32)
    actions[:, 9] = actions[:, 19] = 1.0
    actions[5:, 9] = actions[5:, 19] = 0.0
    actions[5, :3] = (0.0, 0.0, 0.9)
    actions[5, 10:13] = (0.2, 0.0, 0.9)

    target, step = grasp_target(actions)
    normalized = normalize_target_span(target, span=0.16)

    assert step == 5
    np.testing.assert_allclose(normalized.reshape(2, 3).mean(0), (0.1, 0.0, 0.9))
    np.testing.assert_allclose(
        np.linalg.norm(normalized[3:6] - normalized[0:3]), 0.16, atol=1e-6
    )


def test_spatial_grid_and_warp_are_bounded_and_keep_start_fixed() -> None:
    target = np.asarray([0.0, 0.0, 0.9, 0.2, 0.0, 0.9], dtype=np.float32)
    hypotheses, rows = spatial_target_grid(
        target,
        along=(0.0, 0.02),
        perpendicular=(-0.01, 0.01),
    )
    source = np.zeros((8, 16), dtype=np.float32)
    source[:, 3] = source[:, 11] = 1.0
    source[4:, 7] = source[4:, 15] = 0.0
    source[4, :3] = target[:3]
    source[4, 8:11] = target[3:]

    warped = warp_ee_trajectory(source, target, hypotheses[3], grasp_step=4)

    assert hypotheses.shape == (4, 6)
    assert len(rows) == 4
    np.testing.assert_array_equal(warped[0], source[0])
    np.testing.assert_allclose(warped[4, :3], hypotheses[3, :3], atol=1e-6)
    np.testing.assert_allclose(warped[4, 8:11], hypotheses[3, 3:], atol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(warped[:, 3:7], axis=-1), 1.0)
    np.testing.assert_allclose(np.linalg.norm(warped[:, 11:15], axis=-1), 1.0)


def test_native_alignment_and_preservation_gate() -> None:
    assert hasattr(grasp_geometry, "native_target_alignment")
    assert hasattr(grasp_geometry, "calibrate_preservation_gate")
    target = np.asarray([0.0, 0.0, 0.9, 0.2, 0.0, 0.9], dtype=np.float32)
    actions = np.zeros((6, 16), dtype=np.float32)
    actions[:, 3] = actions[:, 11] = 1.0
    actions[:, 7] = actions[:, 15] = 1.0
    actions[3:, 7] = actions[3:, 15] = 0.0
    actions[3, :3] = target[:3]
    actions[3, 8:11] = target[3:]

    aligned = grasp_geometry.native_target_alignment(actions, target)
    shifted = grasp_geometry.native_target_alignment(
        actions, target + np.asarray([0.1, 0, 0, 0.1, 0, 0])
    )
    gate = grasp_geometry.calibrate_preservation_gate(
        np.asarray([aligned, shifted, 0.2]),
        native_success=np.asarray([True, False, True]),
        correction_success=np.asarray([False, True, True]),
    )

    assert aligned < 1e-6
    assert shifted > 0.09
    assert bool(gate.choose_correction(np.asarray([aligned]))) is False
    assert bool(gate.choose_correction(np.asarray([shifted]))) is True
    assert gate.development_success == 3


def test_relational_gate_corrects_early_native_grasp_and_preserves_late_grasp() -> None:
    assert hasattr(grasp_geometry, "native_target_relations")
    assert hasattr(grasp_geometry, "calibrate_early_grasp_gate")
    target = np.asarray([0.0, 0.0, 0.9, 0.2, 0.0, 0.9], dtype=np.float32)
    actions = np.zeros((10, 16), dtype=np.float32)
    actions[:, 3] = actions[:, 11] = 1.0
    actions[:, 7] = actions[:, 15] = 1.0
    actions[2:, 7] = actions[2:, 15] = 0.0
    actions[2, 0:3] = target[0:3]
    actions[2, 8:11] = target[3:6]

    relations = grasp_geometry.native_target_relations(actions, target)
    gate = grasp_geometry.calibrate_early_grasp_gate(
        np.asarray([0.2, 0.7, 0.8]),
        native_success=np.asarray([False, True, True]),
        correction_success=np.asarray([True, False, True]),
    )

    assert relations["alignment_cost_m"] < 1e-6
    assert relations["closest_closed_step_fraction"] == 0.2
    assert relations["closed_fraction"] == 0.8
    assert bool(gate.choose_correction(np.asarray([0.2]))) is True
    assert bool(gate.choose_correction(np.asarray([0.7]))) is False
    assert gate.development_success == 3
