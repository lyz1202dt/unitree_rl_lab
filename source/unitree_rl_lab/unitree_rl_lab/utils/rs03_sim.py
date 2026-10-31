"""Opt-in setup for learned RS03 actuators in the mydog task."""

import math
from pathlib import Path
import shutil

from unitree_rl_lab.assets.robots.rs03_actuator import RS03ActuatorNet, RS03ActuatorNetCfg
from unitree_rl_lab.utils.rs03_network import read_contract, sha256


def configure_rs03_network(env_cfg, args, snapshot_dir=None):
    if not args.actuator_network:
        if args.actuator_metadata or args.actuator_effort_limit is not None:
            raise ValueError("--actuator-metadata/--actuator-effort-limit 需要同时指定 --actuator-network")
        return
    if args.task != "mydog":
        raise ValueError("当前 RS03 执行器网络入口只支持 --task mydog")
    if getattr(args, "delay_min", None) is not None or getattr(args, "delay_max", None) is not None:
        raise ValueError("网络已拟合采样响应，不能同时使用 --delay-min/--delay-max 叠加延时")
    network, metadata, _ = read_contract(args.actuator_network, args.actuator_metadata, env_cfg.sim.dt)
    old = env_cfg.scene.robot.actuators["legs"]
    effort_limit = args.actuator_effort_limit if args.actuator_effort_limit is not None else old.effort_limit
    if not isinstance(effort_limit, (float, int)) or not math.isfinite(effort_limit) or effort_limit <= 0:
        raise ValueError("执行器网络的仿真力矩上限必须是正的有限数值")
    if snapshot_dir is not None:
        destination = Path(snapshot_dir)
        destination.mkdir(parents=True, exist_ok=True)
        for source, name in ((network, "rs03_actuator.pt"), (metadata, "rs03_actuator.json")):
            target = destination / name
            if source != target.resolve():
                shutil.copy2(source, target)
        network, metadata = destination / "rs03_actuator.pt", destination / "rs03_actuator.json"
    env_cfg.scene.robot.actuators["legs"] = RS03ActuatorNetCfg(
        joint_names_expr=old.joint_names_expr,
        effort_limit=float(effort_limit), effort_limit_sim=old.effort_limit_sim,
        velocity_limit=old.velocity_limit, velocity_limit_sim=old.velocity_limit_sim,
        armature=old.armature, friction=old.friction,
        dynamic_friction=old.dynamic_friction, viscous_friction=old.viscous_friction,
        network_file=str(network), metadata_file=str(metadata),
        network_sha256=sha256(network), metadata_sha256=sha256(metadata),
        sample_dt=env_cfg.sim.dt,
    )
    print(f"[信息] 已选择 RS03 执行器网络：{network}")
    print(f"[信息] 物理周期={env_cfg.sim.dt:g} s；输入=e(t,t-1,t-2),dq(t,t-1,t-2)；"
          f"直接输出力矩，仿真上限={effort_limit:g} N·m")
    print("[提示] 下一采样力矩预测用于紧接着的物理步；这是采样近似，不额外叠加 PD 或随机延时。")
    print("[提示] 仿真力矩上限沿用任务或命令参数，不是从采集数据识别出的硬件上限。")


def report_rs03_network(env):
    for name, actuator in env.unwrapped.scene["robot"].actuators.items():
        if isinstance(actuator, RS03ActuatorNet):
            print(f"[信息] RS03 执行器 {name} 已执行 {actuator.compute_calls} 次物理步推理；"
                  f"网络 SHA256={actuator.cfg.network_sha256}", flush=True)
