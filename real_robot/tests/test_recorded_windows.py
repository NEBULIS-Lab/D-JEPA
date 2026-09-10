import importlib

import numpy as np
import pytest


def test_only_consecutive_cached_frames_with_valid_physical_interval_are_paired():
    module = importlib.import_module("djepa_robot.recorded_windows")
    stamps = np.array([0,1500000000,1600000000,1700000000,1800000000,1900000000])
    pairs = module.cached_transition_pairs(np.array([0,1,2,4,5]), stamps, min_dt=.08, max_dt=.15)
    assert pairs == [(1,2,1), (3,4,4)]


def test_duplicate_cached_frame_identity_is_rejected():
    module = importlib.import_module("djepa_robot.recorded_windows")
    with pytest.raises(ValueError):
        module.cached_transition_pairs(np.array([0,1,1]), np.arange(4), min_dt=.08, max_dt=.15)
