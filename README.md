# mydog 训练与 sim2sim

## 一、训练

1. 打开终端，进入 `unitree_rl_lab` 项目目录并激活环境：

   ```bash
   conda activate env_robotlab
   ```

2. 启动训练：

   ```bash
   ./unitree_rl_lab.sh -t --task mydog --num_envs 4090 --max_iterations 3000
   ```

   `--num_envs` 是并行训练环境数量，`--max_iterations` 是训练迭代次数，可按需要修改。默认无界面；需要显示仿真窗口时，在 `-t` 后加 `--gui`：

   ```bash
   ./unitree_rl_lab.sh -t --gui --task mydog --num_envs 64 --max_iterations 3000
   ```

3. 训练结果保存在 `logs/rsl_rl/mydog/训练目录/`，其中 `model_数字.pt` 是策略检查点，`params/deploy.yaml` 是该次训练的部署配置。可以查看训练目录：

   ```bash
   ls -lt logs/rsl_rl/mydog
   ```

### 容器训练

在项目根目录构建镜像，然后启动挂载了当前项目的容器：

```bash
./docker/build.sh
./docker/run.sh
```

构建时需要使用宿主机的 `127.0.0.1:7890` 代理，可改用 `./docker/build.sh --proxy`。该选项只影响镜像构建阶段。

镜像基于本机的 `isaac-lab-base:latest`；容器内项目位于 `/workspace/unitree_rl_lab`，宿主机的修改和 `logs/` 会实时同步。进入容器后运行：

```bash
/workspace/isaaclab/isaaclab.sh -p scripts/rsl_rl/train.py --task mydog --headless --num_envs 64 --max_iterations 3000
```

容器默认使用项目内的 `DIAN_compressed/DIAN.usd`。该 USD 及其 `configuration/` 文件被 Git 忽略，换机器时需自行放到相同的项目目录。若容器日志由 root 写入，宿主机读取或清理时可能需要调整文件权限。

## 二、sim2sim

1. 在已激活 `env_robotlab` 的项目终端中，将训练检查点导出为 ONNX 策略。将命令中的 `训练目录名` 和 `编号` 改成实际值：

   ```bash
   ./unitree_rl_lab.sh -p --task mydog --num_envs 1 \
     --checkpoint "logs/rsl_rl/mydog/训练目录名/model_编号.pt" --headless --sim-steps 1
   ```

   导出的文件为 `logs/rsl_rl/mydog/训练目录名/exported/policy.onnx`。播放命令会重新写入该目录下的导出策略文件。去掉 `--headless --sim-steps 1` 可以在 Isaac Lab 窗口中持续播放策略。

2. 启动 MuJoCo sim2sim，将命令中的 `训练目录名` 改成相同的目录名：

   ```bash
   ./deploy/mujoco/mydog/run.sh --policy "logs/rsl_rl/mydog/训练目录名/exported/policy.onnx"
   ```

   需要慢放时加 `--speed`，支持 `1`、`0.5`、`0.25`、`0.1` 倍速，默认 `1`。例如以四分之一速度运行：

   ```bash
   ./deploy/mujoco/mydog/run.sh --policy "logs/rsl_rl/mydog/训练目录名/exported/policy.onnx" --speed 0.25
   ```

   倍速控制窗口播放速度；物理步长仍为 5 ms，策略周期仍为 20 ms，CSV 时间使用仿真时间。`0.25` 倍速下，1 秒仿真时间对应约 4 秒实际时间。无界面模式按计算能力运行，不使用播放倍速。

3. 点击 MuJoCo 窗口，打开 Num Lock，使用数字小键盘操作：

   | 按键 | 操作 |
   | --- | --- |
   | 1 | 从趴下状态起立；推理时按下则返回站立姿态 |
   | 0 | 等待起立完成后开始策略推理 |
   | 3 | 清零速度指令，3 秒到过渡姿态，再用 1 秒趴下，最后恢复失能阻尼状态 |
   | 8 / 2 | 前进速度增加 / 减少 0.1 m/s |
   | 7 / 9 | 横向速度增加 / 减少 0.1 m/s |
   | 4 / 6 | 转向速度增加 / 减少 0.1 rad/s |
   | 5 | 将所有速度指令设为零 |
   | −（小键盘减号） | 降低播放速度一档 |
   | +（小键盘加号） | 提高播放速度一档，最高 1 倍 |
   | ×（小键盘乘号） | 恢复 1 倍播放速度 |

   启动和最终趴下姿态由 `PRONE_POS` 定义（J1/J2/J3，单位 rad）：FL、RL 为 `0/-0.9/0.6`，FR、RR 为 `0/0.9/-0.6`。起身、趴下途中共用 `PRESTAND_POS` 过渡姿态：当前 FL、RL 为 `0/0.9/0.5`，FR、RR 为 `0/-0.9/-0.5`。两组角度在 `deploy/mujoco/mydog/simulate.py` 中分别修改。

   按 `1` 后，先用 1 秒到过渡姿态，再用 3 秒起身。站立目标读取模型配套 `params/deploy.yaml` 中的 `default_joint_pos`，与推理使用的默认关节角一致；推理时按 `1`，用 3 秒直接返回站立姿态。按 `3` 后，先用 3 秒到同一过渡姿态，再用 1 秒到趴下姿态，最后进入失能阻尼状态。失能时关节角可能随接触和重力变化。

