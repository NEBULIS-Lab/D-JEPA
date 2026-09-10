"""Read-only recorded RGB bridge, using the official AC example's preprocessing."""

import importlib
from pathlib import Path
import sys

import h5py
import numpy as np
import torch
import torch.nn.functional as F


def read_rgb_frames(path, camera, indices):
    path = Path(path).resolve(strict=True)
    if camera not in ("cam_head", "cam_wrist"):
        raise ValueError("camera must be cam_head or cam_wrist")
    indices = np.asarray(indices)
    if indices.ndim != 1 or not len(indices) or indices.dtype.kind not in "iu":
        raise ValueError("frame indices must be a nonempty integer vector")
    with h5py.File(path, "r") as f:
        required = [f"{camera}/color", f"{camera}/timestamp", "left_arm/timestamp",
                    "left_arm/joint", "left_arm/gripper", "left_arm/action"]
        if any(k not in f for k in required):
            raise ValueError("missing recorded RGB/state/command fields")
        images = f[f"{camera}/color"]
        if images.ndim != 4 or images.shape[-1] != 3 or images.dtype != np.uint8:
            raise ValueError("recorded RGB must be uint8 [T,H,W,3]")
        n = images.shape[0]
        if np.any(indices < 0) or np.any(indices >= n):
            raise ValueError("frame indices outside episode")
        if any(f[k].shape[0] != n for k in required):
            raise ValueError("recorded stream lengths disagree")
        # Individual frame reads preserve arbitrary requested order/duplicates;
        # never materialize the full RGB episode just to select a few frames.
        rgb = np.stack([images[int(i)] for i in indices])
        def rows(key):
            return np.stack([f[key][int(i)] for i in indices])
        joints, gripper, commands = rows("left_arm/joint"), rows("left_arm/gripper"), rows("left_arm/action")
        if joints.shape != (len(indices), 6) or commands.shape != (len(indices), 7):
            raise ValueError("recorded joint/command shape mismatch")
        state = np.concatenate((joints, gripper.reshape(len(indices), 1)), axis=-1)
        if not np.isfinite(state).all() or not np.isfinite(commands).all():
            raise ValueError("nonfinite recorded state/command")
        camera_time, robot_time = rows(f"{camera}/timestamp"), rows("left_arm/timestamp")
        if camera_time.dtype.kind not in "iu" or robot_time.dtype.kind not in "iu":
            raise ValueError("recorded timestamps must be integer nanoseconds")
    return {"rgb": rgb, "frame_indices": indices.astype(np.int64),
            "camera_timestamp_ns": camera_time, "robot_timestamp_ns": robot_time,
            "joint_state": state, "joint_commands": commands,
            "episode_path": np.array(str(path)), "camera": np.array(camera),
            "color_order": np.array("RGB"),
            "recorded_action_semantics": np.array("joint_position_target_plus_gripper")}


def official_ac_transform(checkout):
    """Use the actual AC demo transform, not the video-classification hub transform."""
    checkout = Path(checkout).resolve(strict=True)
    if not (checkout / "app/vjepa_droid/transforms.py").is_file():
        raise FileNotFoundError("official AC transforms absent from checkout")
    for name in ("app", "src"):
        existing = sys.modules.get(name)
        locations = list(getattr(existing, "__path__", [])) if existing else []
        if locations and any(not Path(loc).resolve().is_relative_to(checkout) for loc in locations):
            raise ValueError(f"{name} imported from another checkout; run in a fresh process")
    sys.path.insert(0, str(checkout))
    try:
        module = importlib.import_module("app.vjepa_droid.transforms")
        if not Path(module.__file__).resolve().is_relative_to(checkout):
            raise ValueError("AC transform resolved to another checkout")
        return module.make_transforms(random_horizontal_flip=False,
                                      random_resize_aspect_ratio=(1., 1.),
                                      random_resize_scale=(1., 1.), reprob=0., auto_augment=False,
                                      motion_shift=False, crop_size=256)
    finally:
        sys.path.remove(str(checkout))


def validate_rgb(frames):
    if not isinstance(frames, np.ndarray) or frames.dtype != np.uint8:
        raise ValueError("RGB frames must be uint8, not pre-normalized floats")
    if frames.ndim != 4 or frames.shape[-1] != 3 or min(frames.shape) < 1:
        raise ValueError("RGB frames must be nonempty [N,H,W,3]")


@torch.inference_mode()
def encode_rgb_frames(encoder, frames, transform, *, batch_size=2, device="cpu"):
    validate_rgb(frames)
    if not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("positive integer batch size required")
    encoder.eval()
    outputs = []
    for begin in range(0, len(frames), batch_size):
        clip = transform(frames[begin:begin + batch_size])
        if clip.ndim != 4 or clip.shape[:2] != (3, min(batch_size, len(frames) - begin)):
            raise ValueError("transform must return [3,N,H,W]")
        # Each physical frame becomes its own duplicated two-frame tubelet;
        # neighboring recorded frames must NOT be merged into a single embedding.
        tubelets = clip.permute(1, 0, 2, 3).unsqueeze(2).repeat(1, 1, 2, 1, 1).to(device)
        features = encoder(tubelets)
        if features.ndim != 3 or features.shape[0] != tubelets.shape[0] or not torch.isfinite(features).all():
            raise ValueError("encoder must return finite [N,P,D] latents")
        outputs.append(F.layer_norm(features.float(), (features.shape[-1],)).cpu())
    return torch.cat(outputs)
