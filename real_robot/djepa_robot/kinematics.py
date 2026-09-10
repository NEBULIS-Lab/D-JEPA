"""Offline nominal kinematics only. No SDK device/controller imports."""

from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

from .actions import pose_delta


class URDFChain:
    def __init__(self, path, *, base, tip, joint_names):
        self.path = Path(path).resolve(strict=True)
        root = ET.parse(self.path).getroot()
        by_child = {}
        for node in root.findall("joint"):
            child = node.find("child").attrib["link"]
            if child in by_child:
                raise ValueError("multiple parents for a URDF link")
            by_child[child] = node
        reversed_chain, visited, link = [], set(), tip
        while link != base:
            if link in visited or link not in by_child:
                raise ValueError("tip is not reachable from base, or URDF contains a cycle")
            visited.add(link)
            node = by_child[link]
            reversed_chain.append(node)
            link = node.find("parent").attrib["link"]
        nodes = list(reversed(reversed_chain))
        names = [n.attrib["name"] for n in nodes if n.attrib["type"] != "fixed"]
        if names != list(joint_names) or len(set(names)) != len(names):
            raise ValueError("explicit joint order does not match the URDF chain")
        self.joint_names, self.base, self.tip = names, base, tip
        self.joints, self.lower, self.upper = [], [], []
        for node in nodes:
            kind = node.attrib["type"]
            if kind not in ("fixed", "revolute", "continuous", "prismatic") or node.find("mimic") is not None:
                raise ValueError("unsupported joint type or mimic coupling")
            origin = node.find("origin")
            xyz = np.fromstring(origin.get("xyz", "0 0 0") if origin is not None else "0 0 0", sep=" ")
            rpy = np.fromstring(origin.get("rpy", "0 0 0") if origin is not None else "0 0 0", sep=" ")
            axis_node = node.find("axis")
            axis = np.fromstring(axis_node.get("xyz", "1 0 0") if axis_node is not None else "1 0 0", sep=" ")
            if any(v.shape != (3,) or not np.isfinite(v).all() for v in (xyz, rpy, axis)) or np.linalg.norm(axis) == 0:
                raise ValueError("invalid URDF origin or axis")
            matrix = np.eye(4)
            matrix[:3, :3], matrix[:3, 3] = Rotation.from_euler("xyz", rpy).as_matrix(), xyz
            self.joints.append((kind, matrix, axis / np.linalg.norm(axis)))
            if kind != "fixed":
                limits = node.find("limit")
                if kind == "continuous":
                    lo, hi = -np.inf, np.inf
                elif limits is None:
                    raise ValueError("bounded joint must declare limits")
                else:
                    lo, hi = float(limits.attrib["lower"]), float(limits.attrib["upper"])
                    if not np.isfinite([lo, hi]).all() or lo > hi:
                        raise ValueError("invalid joint limits")
                self.lower.append(lo)
                self.upper.append(hi)

    def _q(self, q):
        q = np.asarray(q, dtype=np.float64)
        if q.ndim != 2 or q.shape[1] != len(self.joint_names) or len(q) == 0 or not np.isfinite(q).all():
            raise ValueError("joint values must be finite [T,number_of_chain_joints]")
        return q

    def forward(self, q):
        q = self._q(q)
        total = np.tile(np.eye(4), (len(q), 1, 1))
        index = 0
        for kind, origin, axis in self.joints:
            total = total @ origin
            if kind == "fixed":
                continue
            motion = np.tile(np.eye(4), (len(q), 1, 1))
            if kind in ("revolute", "continuous"):
                motion[:, :3, :3] = Rotation.from_rotvec(q[:, index, None] * axis).as_matrix()
            else:
                motion[:, :3, 3] = q[:, index, None] * axis
            total = total @ motion
            index += 1
        return total

    def limit_violations(self, q):
        q = self._q(q)
        return (q < np.asarray(self.lower) - 1e-6) | (q > np.asarray(self.upper) + 1e-6)


def gripper_closedness(recorded, *, record_to_m, closed_m, open_m):
    values = np.asarray(recorded, dtype=np.float64)
    if not np.isfinite(values).all() or not np.isfinite([record_to_m, closed_m, open_m]).all():
        raise ValueError("finite gripper values and units required")
    if record_to_m <= 0 or open_m <= closed_m:
        raise ValueError("positive scale and distinct ordered gripper endpoints required")
    widths = values * record_to_m
    if np.any(widths < closed_m - 1e-9) or np.any(widths > open_m + 1e-9):
        raise ValueError("recorded gripper is outside configured physical range; do not silently clip")
    return np.clip((open_m - widths) / (open_m - closed_m), 0, 1)


def motion_arrays(fk, joint_states, joint_commands, gripper_states, gripper_commands, **gripper_config):
    actual, commanded = fk.forward(joint_states), fk.forward(joint_commands)
    if actual.shape != commanded.shape or len(actual) < 2:
        raise ValueError("matching state/command trajectories of at least two samples required")
    actual_g = gripper_closedness(gripper_states, **gripper_config)
    commanded_g = gripper_closedness(gripper_commands, **gripper_config)
    if actual_g.shape != (len(actual),) or commanded_g.shape != actual_g.shape:
        raise ValueError("gripper trajectories must match joint trajectory length")
    def poses(matrix, gripper):
        return np.concatenate((matrix[:, :3, 3], Rotation.from_matrix(matrix[:, :3, :3]).as_euler("xyz"),
                               gripper[:, None]), axis=-1)
    observed_pose, target_pose = poses(actual, actual_g), poses(commanded, commanded_g)
    return {"nominal_flange_pose": observed_pose,
            "nominal_command_target_pose": target_pose,
            "command_target_delta": pose_delta(observed_pose, target_pose),
            "observed_transition_delta": pose_delta(observed_pose[:-1], observed_pose[1:]),
            "state_joint_limit_violations": fk.limit_violations(joint_states),
            "command_joint_limit_violations": fk.limit_violations(joint_commands)}
