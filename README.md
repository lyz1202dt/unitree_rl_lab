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


3. 点击 MuJoCo 窗口，打开 Num Lock，使用数字小键盘操作：

   | 按键 | 操作 |
   | --- | --- |
   | 1 | 从趴下状态起立；推理时按下则返回站立姿态 |
   | 0 | 等待起立完成后开始策略推理 |
   | 8 / 2 | 前进速度增加 / 减少 0.1 m/s |
   | 7 / 9 | 横向速度增加 / 减少 0.1 m/s |
   | 4 / 6 | 转向速度增加 / 减少 0.1 rad/s |
   | 5 | 将所有速度指令设为零 |


