# mydog MuJoCo simulation

This scene uses `DIAN_rl_compressed.urdf` for mass, inertia, joints, visual
meshes and collision meshes. The referenced high-resolution collision meshes
are kept separately from the low-poly visuals. MuJoCo computes a convex hull
for each moving mesh when handling contacts, so its contact geometry may differ
from the original non-convex mesh. The oversized base collision STL is converted
to OBJ without removing triangles to satisfy MuJoCo's STL face limit.
Physics runs at 200 Hz; policy inference runs at 50 Hz.
The scene's home keyframe uses J3 angles of -0.5 rad on FL/RL and +0.5 rad on
FR/RR. The simulator starts instead in the prone pose from
`custom_quad/config/deployment_policy/custom_quad/base.yaml`, with the base
near the floor and passive damping (Kp=0, Kd=0.3). It uses that config's fixed
stand pose and a three-second transition. MuJoCo needs Kp=100, Kd=1 for this
get-up under its 25 Nm torque limit; the hardware fixed gains of 25/0.5 do not
raise this model in the simulation. Policy inference uses the default pose
recorded in the policy's `deploy.yaml` and Kp=40, Kd=0.4.
During training, each environment redraws each leg joint's PD gains on reset:
Kp is uniform in [36, 44] and Kd in [0.32, 0.48]. Isaac Lab play and this
MuJoCo simulator use the nominal 40/0.4 gains.
Actuator command delay is currently 0 ms in training and simulation. It applies
to joint position targets before the PD controller; joint feedback is current.

Activate the training Conda environment, then run from the repository root:

```bash
conda activate env_robotlab
./deploy/mujoco/mydog/run.sh
```

This opens a MuJoCo window with the robot prone and passive. To run an
exported policy:

```bash
./deploy/mujoco/mydog/run.sh \
  --policy logs/rsl_rl/mydog/2026-10-27_21-43-47/exported/policy.onnx \
  --vx 0.2 --vy 0 --yaw 0
```

`--policy` accepts a RSL-RL `model_*.pt` checkpoint or an exported
`policy.onnx`. The matching `params/deploy.yaml` is found from the training
run; use `--deploy PATH` if the policy has been moved. The script uses the
45-element policy observation and 12-element joint position action from the
training task. It maps joint names through `joint_ids_map`, applies action
scale and default offset, and sends clipped PD torque (25 Nm per joint). The
simulator applies Kp=40 and Kd=0.4 for policy control and prints these values
at startup. An older `deploy.yaml` may record different
gains; the simulator reports that difference and still applies 40/0.4. The
example ONNX policy above was trained with 25/0.25, so retrain and export a
new policy for a matched 40/0.4 setup.
The `scene.xml` joint `damping="0"` is intentional: Kd is applied by the
controller in `simulate.py`, so adding it there would count damping twice.
The `--vx`, `--vy` and `--yaw` options set the command that takes effect after
policy start. Click the MuJoCo window and enable Num Lock. Press NumPad 1 to
move from passive/prone to the fixed stand pose over three seconds; press
NumPad 0 after the transition to start policy inference. Pressing NumPad 1
again while the policy runs returns to the fixed stand pose. The numeric
keypad controls are:

| Key | Command change |
| --- | --- |
| 1 | Start standing transition; return to stand from policy |
| 0 | Start policy inference after standing |
| 8 / 2 | Forward speed +0.1 / -0.1 m/s |
| 7 / 9 | Leftward speed +0.1 / -0.1 m/s |
| 4 / 6 | Yaw rate +0.1 / -0.1 rad/s |
| 5 | Set all commands to zero |

The limits match the task's command range: forward `[-1, 1]` m/s, sideways
`[-0.4, 0.4]` m/s and yaw `[-1, 1]` rad/s. Current values are printed in the
terminal. Velocity keys affect the loaded policy only after NumPad 0. Without
`--policy`, the robot can stand but cannot enter inference. The example
checkpoint may still be in training, so stable walking is not guaranteed.

To test a fixed delay, add `--action-delay-ms 5` or `10`. The selected delay is
printed at startup.

After NumPad 0 starts policy inference, the simulator saves the four J3 target
and current joint angles to a CSV under `logs/mujoco/mydog/`. It records every
policy step (50 Hz), including zero velocity commands. Columns contain simulation
time followed by paired target and current angles for FL/FR/RL/RR J3, all in
radians. Targets are policy actions after scale and default offset, before the
PD controller; current angles are measured at that same inference instant,
before the physics step. The CSV is created on the first NumPad 0 press, and
its path is printed then and at shutdown. Use `--j3-csv PATH` to choose a path;
an existing file is never overwritten.

Run without a window to check the passive prone startup (no keypad input):

```bash
./deploy/mujoco/mydog/run.sh --headless --duration 5
```

To regenerate the scene from a revised DIAN model, run:

```bash
python deploy/mujoco/mydog/build_scene.py \
  --source /home/hi/下载/点足总装修订版10-18/DIAN
```

The generator writes `scene.xml` and copies the URDF's visual and collision
meshes into `assets/visual/` and `assets/collision/`. The resulting scene does
not need the original model directory to run.
