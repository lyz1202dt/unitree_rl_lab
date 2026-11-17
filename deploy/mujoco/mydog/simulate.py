"""Run the DIAN/mydog robot in MuJoCo, optionally with a trained policy."""

from __future__ import annotations

import argparse
import ast
import csv
from collections import deque
from datetime import datetime
import queue
import re
import time
from pathlib import Path

import glfw
import mujoco
import numpy as np


HERE = Path(__file__).resolve().parent
SDK_NAMES = [f"{leg}_J{joint}" for leg in ("FL", "FR", "RL", "RR") for joint in (1, 2, 3)]
J3_NAMES = [f"{leg}_J3" for leg in ("FL", "FR", "RL", "RR")]
DEFAULT_POS = {
    "FL_J1": 0.0, "FL_J2": 0.9, "FL_J3": -0.5,
    "FR_J1": 0.0, "FR_J2": -0.9, "FR_J3": 0.5,
    "RL_J1": 0.0, "RL_J2": 0.7, "RL_J3": -0.5,
    "RR_J1": 0.0, "RR_J2": -0.7, "RR_J3": 0.5,
}
DEFAULT_KP = 40.0
DEFAULT_KD = 0.4
TORQUE_LIMIT = 25.0
PRESTAND_POS = {
    "FL_J1": -0.0, "FL_J2": 0.2, "FL_J3": 0.5,
    "FR_J1": 0.0, "FR_J2": -0.2, "FR_J3": -0.5,
    "RL_J1": 0.0, "RL_J2": 0.2, "RL_J3": 0.5,
    "RR_J1": 0.0, "RR_J2": -0.2, "RR_J3": -0.5,
}
# Initial and final prone pose, distinct from the shared transition pose above.
PRONE_POS = {
    "FL_J1": 0.0, "FL_J2": -0.9, "FL_J3": 0.6,
    "FR_J1": 0.0, "FR_J2": 0.9, "FR_J3": -0.6,
    "RL_J1": 0.0, "RL_J2": -0.9, "RL_J3": 0.6,
    "RR_J1": 0.0, "RR_J2": 0.9, "RR_J3": -0.6,
}
PRONE_BASE_HEIGHT = 0.165
PASSIVE_KD = 0.3
# Shared gains for posture transitions and standing hold.
STAND_KP = 65.0
STAND_KD = 1.0
PRESTAND_DURATION = 1.0  # Prone <-> shared transition pose.
GETUP_DURATION = 3.0  # Shared transition pose -> standing.
LIE_DOWN_DURATION = 3.0  # Current pose -> shared transition pose when lying down.
PLAYBACK_SPEEDS = (0.1, 0.25, 0.5, 1.0)
COMMAND_STEP = 0.1
COMMAND_LIMITS = ((-1.0, 1.0), (-0.4, 0.4), (-1.0, 1.0))
KEYPAD_INCREMENTS = {
    glfw.KEY_KP_8: (0, 1),
    glfw.KEY_KP_2: (0, -1),
    glfw.KEY_KP_7: (1, 1),
    glfw.KEY_KP_9: (1, -1),
    glfw.KEY_KP_4: (2, 1),
    glfw.KEY_KP_6: (2, -1),
}
KEYPAD_STOP = {glfw.KEY_KP_5}


class StartupController:
    """Handle passive, get-up, standing, policy and lie-down transitions."""

    def __init__(self, prone: np.ndarray, prestand: np.ndarray, stand: np.ndarray):
        self.prone = prone
        self.prestand = prestand
        self.stand = stand
        self.mode = "passive"
        self.start_pos = prone.copy()
        self.prepose_needed = False
        self.elapsed = 0.0

    def request_stand(self, current_pos: np.ndarray) -> bool:
        if self.mode not in ("passive", "policy"):
            return False
        self.prepose_needed = self.mode == "passive"
        self.start_pos = current_pos.copy()
        self.elapsed = 0.0
        self.mode = "getup"
        return True

    def getup_target(self, dt: float) -> np.ndarray:
        prestand_duration = PRESTAND_DURATION if self.prepose_needed else 0.0
        total_duration = prestand_duration + GETUP_DURATION
        self.elapsed = min(total_duration, self.elapsed + dt)
        if self.elapsed >= total_duration - 1e-9:
            self.elapsed = total_duration
        if self.prepose_needed and self.elapsed <= prestand_duration:
            alpha = self.elapsed / prestand_duration
            target = (1.0 - alpha) * self.start_pos + alpha * self.prestand
        else:
            alpha = (self.elapsed - prestand_duration) / GETUP_DURATION
            start_pos = self.prestand if self.prepose_needed else self.start_pos
            target = (1.0 - alpha) * start_pos + alpha * self.stand
        if self.elapsed >= total_duration:
            self.mode = "standing"
        return target

    def request_policy(self, policy_loaded: bool) -> bool:
        if self.mode != "standing" or not policy_loaded:
            return False
        self.mode = "policy"
        return True

    def request_lie_down(self, current_pos: np.ndarray) -> bool:
        if self.mode not in ("getup", "standing", "policy"):
            return False
        self.start_pos = current_pos.copy()
        self.elapsed = 0.0
        self.mode = "lie_down"
        return True

    def get_lie_down_target(self, dt: float) -> np.ndarray:
        total_duration = LIE_DOWN_DURATION + PRESTAND_DURATION
        self.elapsed = min(total_duration, self.elapsed + dt)
        if self.elapsed >= total_duration - 1e-9:
            self.elapsed = total_duration
        if self.elapsed <= LIE_DOWN_DURATION:
            alpha = self.elapsed / LIE_DOWN_DURATION
            target = (1.0 - alpha) * self.start_pos + alpha * self.prestand
        else:
            alpha = (self.elapsed - LIE_DOWN_DURATION) / PRESTAND_DURATION
            target = (1.0 - alpha) * self.prestand + alpha * self.prone
        if self.elapsed >= total_duration:
            self.mode = "passive"
        return target


def update_command(command: np.ndarray, keycode: int) -> bool:
    """Apply a keypad press to [forward, lateral, yaw] velocity commands."""
    if keycode in KEYPAD_STOP:
        if not np.any(command):
            return False
        command[:] = 0.0
        return True
    if keycode not in KEYPAD_INCREMENTS:
        return False
    axis, direction = KEYPAD_INCREMENTS[keycode]
    lower, upper = COMMAND_LIMITS[axis]
    value = float(np.clip(round(float(command[axis]) + direction * COMMAND_STEP, 2), lower, upper))
    if value == command[axis]:
        return False
    command[axis] = value
    return True


def deploy_list(config: str, key: str) -> list:
    match = re.search(rf"(?m)^{re.escape(key)}:\s*(\[[^\]]*\])", config)
    if match is None:
        raise ValueError(f"Missing {key} in deploy.yaml")
    return ast.literal_eval(match.group(1))


def load_deploy_config(path: Path) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    config = path.read_text()
    ids = deploy_list(config, "joint_ids_map")
    if sorted(ids) != list(range(12)):
        raise ValueError("deploy.yaml joint_ids_map must contain each of 0..11 once")
    # Each entry maps an Isaac/actor joint position to its index in SDK_NAMES.
    names = [SDK_NAMES[sdk_index] for sdk_index in ids]
    defaults = np.asarray(deploy_list(config, "default_joint_pos"), dtype=np.float64)
    stiffness = np.asarray(deploy_list(config, "stiffness"), dtype=np.float64)[ids]
    damping = np.asarray(deploy_list(config, "damping"), dtype=np.float64)[ids]
    actions = config.split("\nactions:\n", 1)[1].split("\nobservations:\n", 1)[0]
    scales = np.asarray(deploy_list(actions, "    scale"), dtype=np.float64)
    offsets = np.asarray(deploy_list(actions, "    offset"), dtype=np.float64)
    if any(array.shape != (12,) for array in (defaults, stiffness, damping, scales)):
        raise ValueError("Expected 12 joints in deploy.yaml")
    if offsets.shape != (12,) or not np.allclose(offsets, defaults):
        raise ValueError("This simulator expects action offsets to equal default joint positions")
    return names, defaults, stiffness, damping, scales


def find_deploy_config(policy_path: Path) -> Path:
    for directory in (policy_path.parent, policy_path.parent.parent):
        candidate = directory / "params" / "deploy.yaml"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("deploy.yaml not found near policy; supply --deploy")


def load_policy(path: Path):
    if path.suffix == ".onnx":
        import onnxruntime as ort

        ort.disable_telemetry_events()
        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        if len(session.get_inputs()) != 1:
            raise ValueError("Expected one ONNX policy input")
        input_name = session.get_inputs()[0].name

        def infer(observation: np.ndarray) -> np.ndarray:
            result = session.run(None, {input_name: observation[None].astype(np.float32)})[0]
            return np.asarray(result).reshape(-1).astype(np.float64)

        return infer

    if path.suffix == ".pt":
        import torch

        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        state = checkpoint["model_state_dict"]
        if any("normalizer" in name for name in state):
            raise ValueError("This checkpoint has an observation normalizer; export policy.onnx and use that")
        layers = []
        for index in (0, 2, 4, 6):
            weights = state[f"actor.{index}.weight"].numpy().astype(np.float64)
            bias = state[f"actor.{index}.bias"].numpy().astype(np.float64)
            layers.append((weights, bias))
        if layers[0][0].shape[1] != 45 or layers[-1][0].shape[0] != 12:
            raise ValueError("Expected a 45-input, 12-output mydog policy")

        def infer(observation: np.ndarray) -> np.ndarray:
            value = observation
            for weights, bias in layers[:-1]:
                value = weights @ value + bias
                value = np.where(value > 0, value, np.expm1(value))  # ELU
            weights, bias = layers[-1]
            return weights @ value + bias

        return infer

    raise ValueError("Policy must be an RSL-RL .pt checkpoint or exported .onnx file")


def make_observation(data: mujoco.MjData, qpos_ids: np.ndarray, qvel_ids: np.ndarray,
                     defaults: np.ndarray, command: np.ndarray,
                     last_action: np.ndarray) -> np.ndarray:
    rotation = np.empty(9, dtype=np.float64)
    mujoco.mju_quat2Mat(rotation, data.qpos[3:7])
    gravity_body = rotation.reshape(3, 3).T @ np.array([0.0, 0.0, -1.0])
    gyro_body = data.sensor("base_gyro").data.copy()
    observation = np.concatenate((
        gyro_body * 0.2,
        gravity_body,
        command,
        data.qpos[qpos_ids] - defaults,
        data.qvel[qvel_ids] * 0.05,
        last_action,
    ))
    if observation.shape != (45,):
        raise ValueError(f"Expected 45 policy observations, got {observation.shape}")
    return np.clip(observation, -100.0, 100.0)


def run(args: argparse.Namespace) -> None:
    model = mujoco.MjModel.from_xml_path(str(args.scene))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    if model.nu != 12 or model.nq != 19:
        raise ValueError(f"Unexpected model dimensions: nq={model.nq}, nu={model.nu}")
    if args.policy or args.deploy:
        deploy_path = args.deploy or find_deploy_config(args.policy)
        names, defaults, recorded_kp, recorded_kd, scales = load_deploy_config(deploy_path)
        policy = load_policy(args.policy) if args.policy else None
        if args.policy:
            print(f"Policy: {args.policy}")
        print(f"Deploy config: {deploy_path}")
        if not np.allclose(recorded_kp, DEFAULT_KP) or not np.allclose(recorded_kd, DEFAULT_KD):
            print(
                f"Note: deploy.yaml records Kp={recorded_kp.tolist()}, Kd={recorded_kd.tolist()}; "
                "using the current simulation gains instead."
            )
    else:
        names = SDK_NAMES
        defaults = np.array([DEFAULT_POS[name] for name in names])
        scales = np.full(12, 0.25)
        policy = None
    stiffness = np.full(12, DEFAULT_KP)
    damping = np.full(12, DEFAULT_KD)
    if policy is None:
        print("No policy supplied; NumPad 1 can still raise the robot into standing mode")
    print(f"Startup gains: passive Kp=0, Kd={PASSIVE_KD}; stand Kp={STAND_KP}, Kd={STAND_KD}")
    print(f"Policy joint PD gains: Kp={DEFAULT_KP}, Kd={DEFAULT_KD}")

    joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in names]
    motor_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{name}_motor") for name in names]
    if -1 in joint_ids or -1 in motor_ids:
        raise ValueError("A policy joint or motor is missing from the MuJoCo scene")
    qpos_ids = model.jnt_qposadr[joint_ids]
    qvel_ids = model.jnt_dofadr[joint_ids]
    motor_ids = np.asarray(motor_ids)
    prone = np.array([PRONE_POS[name] for name in names])
    prestand = np.array([PRESTAND_POS[name] for name in names])
    # Reuse the policy defaults in actor joint order, including deploy.yaml overrides.
    stand = defaults.copy()
    joint_ranges = model.jnt_range[joint_ids]
    for pose_name, pose in (("prone", prone), ("pre-stand", prestand), ("stand", stand), ("policy default", defaults)):
        if np.any(pose < joint_ranges[:, 0]) or np.any(pose > joint_ranges[:, 1]):
            raise ValueError(f"The {pose_name} pose exceeds the scene's joint limits")
    data.qpos[2] = PRONE_BASE_HEIGHT
    data.qpos[qpos_ids] = prone
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    startup = StartupController(prone, prestand, stand)
    print("Startup state: prone/passive. NumPad 1: stand; after standing, NumPad 0: start policy; "
          "NumPad 3: lie down and return to passive.")

    command = np.array([args.vx, args.vy, args.yaw], dtype=np.float64)
    last_action = np.zeros(12, dtype=np.float64)
    target = defaults.copy()
    delay_ms = args.action_delay_ms
    delay_steps = delay_ms // 5
    # Match Isaac Lab's DelayBuffer warm-up: before enough samples exist, use
    # the oldest available target rather than a synthetic pre-reset command.
    target_history = deque(maxlen=delay_steps + 1)
    print(f"Action delay: {delay_ms} ms ({delay_steps} physics steps)")
    decimation = round(0.02 / model.opt.timestep)
    if not np.isclose(decimation * model.opt.timestep, 0.02):
        raise ValueError("Physics timestep must divide the 0.02 s policy period")
    if not np.isclose(model.opt.timestep, 0.005):
        raise ValueError("Action delay settings require a 0.005 s physics timestep")
    total_steps = int(round(args.duration / model.opt.timestep)) if args.duration > 0 else None
    if args.headless and total_steps is None:
        raise ValueError("--headless needs a positive --duration")
    if args.j3_csv is not None and policy is None:
        raise ValueError("--j3-csv requires --policy")

    csv_file = None
    csv_writer = None
    csv_path = None
    j3_indices = None
    logged_rows = 0
    policy_step_number = 0
    if policy is not None:
        j3_indices = [names.index(name) for name in J3_NAMES]

    def step() -> None:
        nonlocal last_action, target, logged_rows, policy_step_number
        if startup.mode == "passive":
            torques = -PASSIVE_KD * data.qvel[qvel_ids]
        elif startup.mode in ("getup", "standing", "lie_down"):
            if startup.mode == "getup":
                target = startup.getup_target(model.opt.timestep)
                if startup.mode == "standing":
                    print("Standing pose reached; NumPad 0 starts policy inference.", flush=True)
            elif startup.mode == "lie_down":
                target = startup.get_lie_down_target(model.opt.timestep)
                if startup.mode == "passive":
                    print("Lie-down transition finished; passive damping active.", flush=True)
            else:
                target = stand
            if startup.mode == "passive":
                torques = -PASSIVE_KD * data.qvel[qvel_ids]
            else:
                torques = STAND_KP * (target - data.qpos[qpos_ids]) - STAND_KD * data.qvel[qvel_ids]
        else:
            if policy is not None and policy_step_number % decimation == 0:
                obs = make_observation(data, qpos_ids, qvel_ids, defaults, command, last_action)
                action = np.asarray(policy(obs)).reshape(-1)
                if action.shape != (12,) or not np.isfinite(action).all():
                    raise ValueError("Policy must return 12 finite actions")
                target = np.clip(defaults + scales * action, -100.0, 100.0)
                last_action = action
                if csv_writer is not None:
                    row = [data.time]
                    for index in j3_indices:
                        row.extend((target[index], data.qpos[qpos_ids[index]]))
                    csv_writer.writerow(row)
                    csv_file.flush()
                    logged_rows += 1
            policy_step_number += 1
            target_history.append(target.copy())
            delayed_target = target_history[0]
            torques = stiffness * (delayed_target - data.qpos[qpos_ids]) - damping * data.qvel[qvel_ids]
        data.ctrl[motor_ids] = np.clip(torques, -TORQUE_LIMIT, TORQUE_LIMIT)
        mujoco.mj_step(model, data)

    try:
        if args.headless:
            for _ in range(total_steps):
                step()
        else:
            from mujoco import viewer

            speed_index = PLAYBACK_SPEEDS.index(args.speed)
            key_events: queue.SimpleQueue[int] = queue.SimpleQueue()
            print("Keypad: 1 stand, 0 policy, 3 lie down, 8/2 forward, 7/9 lateral, 4/6 yaw, 5 stops. "
                  "- slower, + faster, * normal speed. Enable Num Lock.", flush=True)
            print(f"Playback speed: {PLAYBACK_SPEEDS[speed_index]:g}x", flush=True)
            with viewer.launch_passive(model, data, key_callback=key_events.put) as viewer_handle:
                viewer_handle.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
                viewer_handle.cam.trackbodyid = model.body("base_link").id
                viewer_handle.cam.distance = 1.7
                viewer_handle.cam.elevation = -15
                index = 0
                while viewer_handle.is_running() and (total_steps is None or index < total_steps):
                    started = time.perf_counter()
                    while True:
                        try:
                            keycode = key_events.get_nowait()
                        except queue.Empty:
                            break
                        if keycode in (glfw.KEY_KP_SUBTRACT, glfw.KEY_KP_ADD, glfw.KEY_KP_MULTIPLY):
                            previous_speed_index = speed_index
                            if keycode == glfw.KEY_KP_SUBTRACT:
                                speed_index = max(0, speed_index - 1)
                            elif keycode == glfw.KEY_KP_ADD:
                                speed_index = min(len(PLAYBACK_SPEEDS) - 1, speed_index + 1)
                            else:
                                speed_index = len(PLAYBACK_SPEEDS) - 1
                            if speed_index != previous_speed_index:
                                print(f"Playback speed: {PLAYBACK_SPEEDS[speed_index]:g}x", flush=True)
                        elif keycode == glfw.KEY_KP_1:
                            if startup.request_stand(data.qpos[qpos_ids]):
                                target_history.clear()
                                last_action[:] = 0.0
                                policy_step_number = 0
                                if startup.prepose_needed:
                                    print(f"Pre-stand transition started ({PRESTAND_DURATION:g} s), "
                                          f"then standing ({GETUP_DURATION:g} s).", flush=True)
                                else:
                                    print(f"Standing transition started ({GETUP_DURATION:g} s).", flush=True)
                        elif keycode == glfw.KEY_KP_3:
                            if startup.request_lie_down(data.qpos[qpos_ids]):
                                target_history.clear()
                                last_action[:] = 0.0
                                policy_step_number = 0
                                command[:] = 0.0
                                print(f"Lie-down started: transition pose ({LIE_DOWN_DURATION:g} s), "
                                      f"then prone ({PRESTAND_DURATION:g} s); "
                                      "then passive damping.", flush=True)
                        elif keycode == glfw.KEY_KP_0:
                            if startup.request_policy(policy is not None):
                                target = stand.copy()
                                target_history.clear()
                                last_action[:] = 0.0
                                policy_step_number = 0
                                if csv_file is None:
                                    csv_path = args.j3_csv or (
                                        HERE.parents[2] / "logs" / "mujoco" / "mydog"
                                        / f"j3_angles_{datetime.now():%Y%m%d_%H%M%S_%f}.csv"
                                    )
                                    csv_path.parent.mkdir(parents=True, exist_ok=True)
                                    csv_file = csv_path.open("x", newline="", encoding="utf-8")
                                    csv_writer = csv.writer(csv_file)
                                    header = ["sim_time_s"]
                                    for name in J3_NAMES:
                                        header.extend((f"{name}_target_rad", f"{name}_current_rad"))
                                    csv_writer.writerow(header)
                                    print(f"J3 angle CSV: {csv_path.resolve()}", flush=True)
                                print("Policy inference started.", flush=True)
                            elif policy is None:
                                print("Load --policy to start inference.", flush=True)
                            elif startup.mode != "policy":
                                print("Wait until the standing transition finishes, then press NumPad 0.", flush=True)
                        elif startup.mode == "policy" and update_command(command, keycode):
                            print(f"Command: vx={command[0]:+.1f}, vy={command[1]:+.1f}, yaw={command[2]:+.1f}", flush=True)
                    step()
                    viewer_handle.sync()
                    index += 1
                    # Slow wall-clock playback without changing physics or policy timing.
                    frame_duration = model.opt.timestep / PLAYBACK_SPEEDS[speed_index]
                    time.sleep(max(0.0, frame_duration - (time.perf_counter() - started)))
    finally:
        if csv_file is not None:
            csv_file.close()
            print(f"Saved {logged_rows} J3 angle rows: {csv_path.resolve()}")

    print(f"Simulation time: {data.time:.2f} s; base xyz: {data.qpos[:3].round(3)}; contacts: {data.ncon}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, default=HERE / "scene.xml")
    parser.add_argument("--policy", type=Path, help="RSL-RL model_*.pt checkpoint or exported policy.onnx")
    parser.add_argument("--deploy", type=Path, help="Associated params/deploy.yaml; found automatically for run checkpoints")
    parser.add_argument("--vx", type=float, default=0.0, help="Commanded forward speed (m/s)")
    parser.add_argument("--vy", type=float, default=0.0, help="Commanded sideways speed (m/s)")
    parser.add_argument("--yaw", type=float, default=0.0, help="Commanded yaw rate (rad/s)")
    parser.add_argument("--action-delay-ms", type=int, choices=(0, 5, 10), default=0,
                        help="Actuator command delay in ms (default: 0)")
    parser.add_argument("--j3-csv", type=Path,
                        help="CSV output path for J3 target and current angles during policy inference; defaults to logs/mujoco/mydog/")
    parser.add_argument("--speed", type=float, choices=PLAYBACK_SPEEDS, default=1.0,
                        help="GUI playback speed multiplier; physics/policy timing stays fixed (default: 1)")
    parser.add_argument("--duration", type=float, default=0.0,
                        help="Simulation duration in seconds; 0 means until GUI closes")
    parser.add_argument("--headless", action="store_true")
    try:
        run(parser.parse_args())
    except KeyboardInterrupt:
        print("\nSimulation interrupted.")
