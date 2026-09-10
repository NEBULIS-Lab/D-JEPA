import importlib

import numpy as np


def test_lag_scan_uses_common_valid_windows_and_recovers_shift():
    module = importlib.import_module("djepa_robot.motion_audit")
    rng = np.random.default_rng(9)
    commands = rng.normal(size=(20, 6))
    observed = np.concatenate((np.zeros((1,6)), commands[:-1]))
    timestamps = np.arange(20, dtype=np.int64) * 100000000
    result = module.tracking_scan(observed, commands, timestamps, max_lag=3, min_dt=.08, max_dt=.15)
    assert result["best_descriptive_lag_steps"] == 1
    assert result["joint_mae_rad_by_lag"][1] == 0
    assert result["common_window_count"] == 17
    assert result["identified_physical_delay"] is False


def test_lag_scan_excludes_gaps_in_all_compared_offsets():
    module = importlib.import_module("djepa_robot.motion_audit")
    stamps = np.array([0, 100000000, 200000000, 1200000000, 1300000000], dtype=np.int64)
    result = module.tracking_scan(np.zeros((5,6)), np.zeros((5,6)), stamps,
                                  max_lag=2, min_dt=.08, max_dt=.15)
    assert result["common_window_count"] == 1
