import importlib
import json

import h5py
import numpy as np
import pytest


def recording(path, *, image_rows=3, commands=True):
    stamps = np.array([1780643630000000000, 1780643631500000000,
                       1780643631600000000], dtype=np.int64)
    with h5py.File(path, "w") as f:
        f["left_arm/joint"] = np.zeros((3, 6))
        f["left_arm/gripper"] = np.zeros(3)
        f["left_arm/timestamp"] = stamps
        if commands:
            f["left_arm/action"] = np.ones((3, 7)) * 0.01
        for camera in ("cam_head", "cam_wrist"):
            f.create_dataset(f"{camera}/color", (image_rows, 8, 8, 3), dtype="uint8")
            f[f"{camera}/timestamp"] = stamps + 1000


def test_retains_commands_and_actual_timestamps(tmp_path):
    data = importlib.import_module("djepa_robot.data")
    path = tmp_path / "0.hdf5"
    recording(path)
    episode = data.read_episode(path)
    assert episode["action_semantics"] == "joint_position_target_plus_gripper"
    assert not np.array_equal(episode["commands"], episode["state"])
    assert episode["timestamp_ns"].dtype == np.int64
    np.testing.assert_allclose(episode["time_s"], [0, 1.5, 1.6])
    assert episode["image_shapes"]["cam_head"] == (3, 8, 8, 3)
    assert "images" not in episode
    report = data.summarize_episode(episode, nominal_fps=10)
    assert report["gap_count"] == 1
    assert report["dt_max_s"] == pytest.approx(1.5)
    assert report["camera_skew_max_s"] == pytest.approx(1e-6)


@pytest.mark.parametrize("kwargs", [{"commands": False}, {"image_rows": 2}])
def test_rejects_missing_commands_and_misaligned_streams(tmp_path, kwargs):
    data = importlib.import_module("djepa_robot.data")
    path = tmp_path / "0.hdf5"
    recording(path, **kwargs)
    with pytest.raises(ValueError):
        data.read_episode(path)


def test_rejects_nonmonotonic_time(tmp_path):
    data = importlib.import_module("djepa_robot.data")
    path = tmp_path / "0.hdf5"
    recording(path)
    with h5py.File(path, "a") as f:
        f["left_arm/timestamp"][1] = f["left_arm/timestamp"][0]
    with pytest.raises(ValueError, match="timestamp"):
        data.read_episode(path)


def test_audit_detects_metadata_disagreement(tmp_path):
    data = importlib.import_module("djepa_robot.data")
    raw, lerobot = tmp_path / "raw", tmp_path / "lerobot"
    raw.mkdir()
    (lerobot / "meta").mkdir(parents=True)
    recording(raw / "0.hdf5")
    (lerobot / "meta/info.json").write_text(json.dumps(
        {"total_episodes": 1, "total_frames": 4, "fps": 10}))
    result = data.audit_dataset(lerobot, raw)
    assert result["valid_episodes"] == 1
    assert not result["metadata_counts_match"]
    assert result["raw_frames"] == 3


def test_windows_do_not_cross_recording_gaps(tmp_path):
    data = importlib.import_module("djepa_robot.data")
    path = tmp_path / "0.hdf5"
    recording(path)
    ep = data.read_episode(path)
    assert data.valid_window_starts(ep["timestamp_ns"], horizon=1,
                                    min_dt=0.08, max_dt=0.15).tolist() == [1]
