"""Torch-only contract and history handling for custom_quad actuator exports."""

import hashlib
import json
import math
from pathlib import Path

import torch


INPUTS = ["position_error_t", "position_error_t_minus_1", "position_error_t_minus_2",
          "velocity_t", "velocity_t_minus_1", "velocity_t_minus_2"]
JOINTS = [f"{leg}_j{j}" for leg in ("FL", "FR", "RL", "RR") for j in (1, 2, 3)]


def read_contract(network_file, metadata_file, simulation_dt):
    network = Path(network_file).expanduser().resolve(strict=True)
    metadata = Path(metadata_file).expanduser().resolve(strict=True) if metadata_file else network.with_suffix(".json")
    contract = json.loads(metadata.read_text(encoding="utf-8"))
    required = {
        "inputs": INPUTS,
        "position_error": "clipped_target_q - feedback_q",
        "label": "joint_torque_estimate_at_next_host_sample",
        "label_offset_rows": 1,
        "state": "RLFSMStateRLLocomotion",
    }
    for key, expected in required.items():
        if contract.get(key) != expected:
            raise ValueError(f"执行器网络元数据 {key} 不兼容：{contract.get(key)!r}")
    names = contract.get("joint_order", [])
    if len(names) != 12 or set(names) != set(JOINTS):
        raise ValueError("执行器网络必须来自 mydog 的 12 个关节")
    dt = contract.get("nominal_dt_s")
    if not isinstance(dt, (int, float)) or not math.isfinite(dt) or dt <= 0:
        raise ValueError("执行器网络采样周期无效")
    if not math.isclose(dt, simulation_dt, rel_tol=0, abs_tol=1e-9):
        raise ValueError(f"执行器网络采样周期 {dt} s 与物理步长 {simulation_dt} s 不一致")
    return network, metadata, contract


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class HistoryTorqueModel:
    """Predict torque for the upcoming step, with no additional delay queue.

    The CSV label is the next host sample, not an identified instantaneous torque
    or exact drive delay. Holding this prediction over one physics step is an
    explicit approximation that must be checked against real trajectories.
    """

    def __init__(self, network_file, num_envs, num_joints, device):
        self.network = torch.jit.load(str(network_file), map_location=device).eval()
        for parameter in self.network.parameters():
            parameter.requires_grad_(False)
        self.errors = torch.zeros(num_envs, num_joints, 3, device=device)
        self.velocities = torch.zeros_like(self.errors)
        self.fresh = torch.ones(num_envs, dtype=torch.bool, device=device)
        with torch.inference_mode():
            output = self.network(torch.zeros(num_envs * num_joints, 6, device=device))
        if output.shape != (num_envs * num_joints, 1) or not torch.isfinite(output).all():
            raise ValueError("执行器网络必须接受 [N,6] 并返回有限的 [N,1] 力矩")

    def reset(self, env_ids):
        if env_ids is None:
            env_ids = slice(None)
        self.errors[env_ids] = 0
        self.velocities[env_ids] = 0
        self.fresh[env_ids] = True

    def step(self, error, velocity):
        if error.shape != self.errors.shape[:2] or velocity.shape != error.shape:
            raise ValueError("执行器网络输入的环境数或关节数不匹配")
        self.errors = self.errors.roll(1, dims=2)
        self.velocities = self.velocities.roll(1, dims=2)
        self.errors[:, :, 0] = error
        self.velocities[:, :, 0] = velocity
        # No history exists after reset; repeat the first measured state.
        self.errors = torch.where(self.fresh[:, None, None], error[:, :, None], self.errors)
        self.velocities = torch.where(self.fresh[:, None, None], velocity[:, :, None], self.velocities)
        self.fresh[:] = False
        inputs = torch.cat((self.errors, self.velocities), dim=2).reshape(-1, 6)
        with torch.inference_mode():
            return self.network(inputs).reshape(error.shape)
