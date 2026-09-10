import importlib

import numpy as np
import pytest

URDF = '''<robot name="fixture">
<link name="base"/><link name="arm"/><link name="tip"/>
<joint name="turn" type="revolute"><parent link="base"/><child link="arm"/>
<origin xyz="1 0 0" rpy="0 0 0"/><axis xyz="0 0 1"/><limit lower="-3.15" upper="3.15"/></joint>
<joint name="tool" type="fixed"><parent link="arm"/><child link="tip"/><origin xyz="1 0 0"/></joint>
</robot>'''


def chain(tmp_path):
    module = importlib.import_module("djepa_robot.kinematics")
    path = tmp_path / "arm.urdf"
    path.write_text(URDF)
    return module.URDFChain(path, base="base", tip="tip", joint_names=["turn"])


def test_fk_rotates_child_translation_in_joint_frame(tmp_path):
    fk = chain(tmp_path)
    t = fk.forward(np.array([[0.0], [np.pi / 2]]))
    np.testing.assert_allclose(t[:, :3, 3], [[2, 0, 0], [1, 1, 0]], atol=1e-10)
    np.testing.assert_allclose(t[1, :3, :3], [[0,-1,0],[1,0,0],[0,0,1]], atol=1e-10)
    assert fk.limit_violations(np.array([[0.0], [4.0]])).tolist() == [[False], [True]]


def test_wrong_joint_order_and_unreachable_tip_rejected(tmp_path):
    module = importlib.import_module("djepa_robot.kinematics")
    path = tmp_path / "arm.urdf"
    path.write_text(URDF)
    for names, tip in [(["wrong"], "tip"), (["turn"], "absent")]:
        with pytest.raises(ValueError):
            module.URDFChain(path, base="base", tip=tip, joint_names=names)


def test_nonfinite_joint_values_rejected(tmp_path):
    fk = chain(tmp_path)
    with pytest.raises(ValueError):
        fk.forward(np.array([[float("nan")]]))


def test_gripper_units_direction_and_no_silent_clipping():
    module = importlib.import_module("djepa_robot.kinematics")
    actual = module.gripper_closedness(np.array([0, .35, .7]), record_to_m=.1, closed_m=0, open_m=.07)
    np.testing.assert_allclose(actual, [1,.5,0], atol=1e-12)
    with pytest.raises(ValueError, match="range"):
        module.gripper_closedness(np.array([.9]), record_to_m=.1, closed_m=0, open_m=.07)


def test_command_and_observed_displacements_remain_distinct(tmp_path):
    module = importlib.import_module("djepa_robot.kinematics")
    fk = chain(tmp_path)
    q = np.array([[0.], [.1], [.2]])
    commands = np.array([[.3], [.4], [.5]])
    out = module.motion_arrays(fk, q, commands, np.array([.7,.7,.7]), np.array([.7,.7,.7]),
                               record_to_m=.1, closed_m=0, open_m=.07)
    assert not np.allclose(out["command_target_delta"][:-1], out["observed_transition_delta"])
    assert out["command_target_delta"].shape == (3,7)
    assert out["observed_transition_delta"].shape == (2,7)
    np.testing.assert_allclose(out["observed_transition_delta"][:,5], [.1,.1], atol=1e-10)
