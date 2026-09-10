import importlib
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch


def fixture(path):
    with h5py.File(path, "w") as f:
        frames = np.zeros((4, 8, 12, 3), dtype=np.uint8)
        frames[0, ..., 0] = 255
        frames[2, ..., 2] = 255
        f["cam_head/color"] = frames
        f["cam_head/timestamp"] = np.arange(4, dtype=np.int64) * 100000000 + 1000
        f["left_arm/timestamp"] = np.arange(4, dtype=np.int64) * 100000000
        f["left_arm/joint"] = np.zeros((4, 6))
        f["left_arm/gripper"] = np.zeros(4)
        f["left_arm/action"] = np.ones((4, 7))


def test_selective_rgb_preserves_order_channels_and_identity(tmp_path):
    vision = importlib.import_module("djepa_robot.vision")
    path = tmp_path / "episode.hdf5"
    fixture(path)
    selected = vision.read_rgb_frames(path, "cam_head", np.array([2, 0, 2]))
    assert selected["rgb"].shape == (3, 8, 12, 3)
    assert selected["rgb"][0, 0, 0].tolist() == [0, 0, 255]
    assert selected["rgb"][1, 0, 0].tolist() == [255, 0, 0]
    assert selected["frame_indices"].tolist() == [2, 0, 2]
    assert selected["camera_timestamp_ns"].tolist() == [200001000, 1000, 200001000]
    assert "pose" not in selected
    assert not np.array_equal(selected["joint_state"], selected["joint_commands"])


@pytest.mark.parametrize("indices", [[-1], [4], [1.2], []])
def test_invalid_frame_indices_rejected(tmp_path, indices):
    vision = importlib.import_module("djepa_robot.vision")
    path = tmp_path / "episode.hdf5"
    fixture(path)
    with pytest.raises(ValueError):
        vision.read_rgb_frames(path, "cam_head", np.array(indices))


def test_official_ac_preprocessing_is_deterministic_and_rgb_correct(upstream_checkout):
    vision = importlib.import_module("djepa_robot.vision")
    checkout = upstream_checkout
    transform = vision.official_ac_transform(checkout)
    frames = np.zeros((2, 48, 64, 3), dtype=np.uint8)
    frames[..., 0] = 255
    first, second = transform(frames), transform(frames)
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    assert first.shape == (3, 2, 256, 256)
    torch.testing.assert_close(first[:, 0, 0, 0],
                               torch.tensor([(1-.485)/.229, -.456/.224, -.406/.225]), atol=1e-6, rtol=1e-6)


def test_encoding_duplicates_each_frame_instead_of_pairing_neighbors():
    vision = importlib.import_module("djepa_robot.vision")
    frames = np.stack((np.full((8, 8, 3), 10, dtype=np.uint8),
                       np.full((8, 8, 3), 90, dtype=np.uint8)))
    class TinyEncoder(torch.nn.Module):
        def forward(self, x):
            assert x.shape[2] == 2
            assert torch.equal(x[:, :, 0], x[:, :, 1])
            # Return three unequal channels so layer norm has a nontrivial expected value.
            scalar = x.mean(dim=(1, 2, 3, 4))
            return torch.stack((scalar, scalar + 1, scalar + 2), dim=-1)[:, None, :]
    transform = lambda x: torch.tensor(x, dtype=torch.float32).permute(3, 0, 1, 2)
    features = vision.encode_rgb_frames(TinyEncoder(), frames, transform, batch_size=1)
    assert features.shape == (2, 1, 3)
    torch.testing.assert_close(features[0, 0], torch.tensor([-1.2247356, 0, 1.2247356]))


def test_float_rgb_is_rejected_instead_of_double_normalized():
    vision = importlib.import_module("djepa_robot.vision")
    with pytest.raises(ValueError, match="uint8"):
        vision.encode_rgb_frames(torch.nn.Identity(), np.zeros((1, 8, 8, 3)), lambda x: x)
