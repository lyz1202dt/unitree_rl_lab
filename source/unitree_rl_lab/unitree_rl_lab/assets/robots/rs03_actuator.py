"""Explicit learned actuator for the 5 ms custom_quad CSV contract."""

from dataclasses import MISSING

from isaaclab.actuators import IdealPDActuator, IdealPDActuatorCfg
from isaaclab.utils import configclass

from unitree_rl_lab.utils.rs03_network import HistoryTorqueModel, JOINTS


class RS03ActuatorNet(IdealPDActuator):
    """Apply network torque directly, with the task's symmetric effort ceiling.

    Inheriting IdealPDActuator only reuses parameter storage and effort clipping.
    Its PD compute method is never called; there is no second PD or delay stage.
    """

    def __init__(self, cfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        if set(name.lower() for name in self.joint_names) != set(name.lower() for name in JOINTS):
            raise ValueError("RS03 执行器网络要求完整的 mydog 12 关节组")
        self.model = HistoryTorqueModel(cfg.network_file, self._num_envs, self.num_joints, self._device)
        self.target_lower = self._parse_joint_parameter(cfg.target_lower, None)
        self.target_upper = self._parse_joint_parameter(cfg.target_upper, None)
        self.compute_calls = 0

    def reset(self, env_ids):
        self.model.reset(env_ids)

    def compute(self, control_action, joint_pos, joint_vel):
        if control_action.joint_positions is None:
            raise ValueError("RS03 执行器网络只支持位置目标动作")
        target = control_action.joint_positions.clamp(self.target_lower, self.target_upper)
        self.computed_effort = self.model.step(target - joint_pos, joint_vel)
        self.applied_effort = self._clip_effort(self.computed_effort)
        control_action.joint_efforts = self.applied_effort
        control_action.joint_positions = None
        control_action.joint_velocities = None
        self.compute_calls += 1
        return control_action


@configclass
class RS03ActuatorNetCfg(IdealPDActuatorCfg):
    class_type: type = RS03ActuatorNet
    network_file: str = MISSING
    metadata_file: str = MISSING
    network_sha256: str = MISSING
    metadata_sha256: str = MISSING
    sample_dt: float = 0.005
    # The network already represents the sampled host controller and drive.
    stiffness: float = 0.0
    damping: float = 0.0
    # DIAN limits used to clip target_q in the source custom_quad CSV.
    target_lower: dict[str, float] = {".*_J1": -1.0472, ".*_J2": -1.5708, ".*_J3": -0.8552}
    target_upper: dict[str, float] = {".*_J1": 1.0472, ".*_J2": 1.5708, ".*_J3": 0.8552}
