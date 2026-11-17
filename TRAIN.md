# 容器训练：mydog

以下命令均从项目根目录执行。需要本机已有 `isaac-lab-base:latest` 镜像、可供 Docker 使用的 NVIDIA GPU，以及 `/home/lyz/Project/AT_robot_lab/source/robot_lab/data/Robots/03dog` 下的 URDF 和网格文件。容器通过 `docker/.env.base` 中的 `HOST_DOG03_ASSET_PATH` 只读挂载该目录。

## 1. 构建镜像

```bash
./docker/build.sh
```

如需在构建阶段使用宿主机 `127.0.0.1:7890` 代理，改用：

```bash
./docker/build.sh --proxy
```

构建得到 `unitree-rl-lab:latest`。代理选项只影响镜像构建，不会给训练容器设置代理。

## 2. 进入容器并训练

```bash
./docker/run.sh
```

容器会把当前项目挂载到 `/workspace/unitree_rl_lab`，并以该目录为工作目录。在容器 shell 中启动无界面训练：

```bash
/workspace/isaaclab/isaaclab.sh -p scripts/rsl_rl/train.py \
  --task mydog --headless --num_envs 64 --max_iterations 3000
```

也可以从宿主机用一条命令直接启动，结束后容器自动删除：

```bash
./docker/run.sh -lc '/workspace/isaaclab/isaaclab.sh -p scripts/rsl_rl/train.py --task mydog --headless --num_envs 64 --max_iterations 3000'
```

`--num_envs` 控制并行环境数；6 GB 显存可先从 `64` 开始，再按显存使用情况调整。`--max_iterations` 控制训练迭代数。容器内使用 Isaac Lab 的 Python，不需要激活宿主机 Conda 环境。

## 3. 查看结果与续训

项目目录是实时挂载的，训练检查点会保留在宿主机：

```bash
ls -lt logs/rsl_rl/mydog
```

每次运行对应 `logs/rsl_rl/mydog/日期时间/`，其中 `model_数字.pt` 是检查点，`params/deploy.yaml` 是部署配置。需要从指定检查点继续训练时，在容器 shell 中运行下面的命令，并把 `RUN_DIR`、`model_数字.pt` 替换成实际名称：

```bash
/workspace/isaaclab/isaaclab.sh -p scripts/rsl_rl/train.py \
  --task mydog --headless --num_envs 64 --max_iterations 3000 \
  --resume --load_run RUN_DIR --checkpoint model_数字.pt
```

续训会读取旧检查点，并创建新的时间戳目录保存本次结果。切换到 03dog 后应重新训练，不要续训旧 DIAN 检查点。`mydog` 使用 `DOG03_URDF_PATH` 指向容器内 `/workspace/03dog/urdf/03dog.urdf`；非容器运行时默认读取上述 `AT_robot_lab` 路径，也可通过同名环境变量覆盖。


## 4. 导出 ONNX 策略

先在宿主机查看要导出的检查点，并把下面命令中的 `RUN_DIR` 和 `model_数字.pt` 换成实际名称：

```bash
ls logs/rsl_rl/mydog/RUN_DIR/model_*.pt
```

在容器 shell 中运行：

```bash
/workspace/isaaclab/isaaclab.sh -p scripts/rsl_rl/play.py \
  --task mydog --checkpoint logs/rsl_rl/mydog/RUN_DIR/model_数字.pt \
  --headless --num_envs 1 --sim-steps 1
```

也可以在宿主机直接运行，命令结束后容器自动删除：

```bash
./docker/run.sh -lc '/workspace/isaaclab/isaaclab.sh -p scripts/rsl_rl/play.py --task mydog --checkpoint logs/rsl_rl/mydog/RUN_DIR/model_数字.pt --headless --num_envs 1 --sim-steps 1'
```

`play.py` 会加载指定检查点，启动一次无界面仿真并导出策略。ONNX 文件位于 `logs/rsl_rl/mydog/RUN_DIR/exported/policy.onnx`；同目录还会生成 TorchScript 文件 `policy.pt`。这些文件通过项目挂载保留在宿主机，重复运行会覆盖对应导出文件。可用以下命令确认：

```bash
ls -lh logs/rsl_rl/mydog/RUN_DIR/exported/
```