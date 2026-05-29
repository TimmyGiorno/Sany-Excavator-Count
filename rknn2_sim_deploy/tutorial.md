# 基于 Docker 的 Windows RKNN 仿真环境部署操作指南

### 0. 相关链接

* 相关教程：https://doc.embedfire.com/linux/rk356x/Ai/zh/latest/lubancat_ai/env/toolkit2.html
* 相关教程：https://www.cnblogs.com/ttkwzyttk/p/19541388
* RKNN Toolkit 官方库：https://github.com/airockchip/rknn-toolkit2

### 1. WSL 环境配置与排查

注：为了使用 Docker，请自行排查电脑是否安装最新版 `WSL`，可使用以下指令排查：

```BASH
wsl --verion
```

如果无法输出 `wsl` 版本，则为电脑未正确安装 `wsl`，使用如下指令安装：

```bash
wsl --install
wsl --update
```

并在管理员的 Terminal 中运行以下指令：

```bash
bcdedit /set hypervisorlaunchtype auto
```

另注：由于国内 Docker 镜像源很多都已经挂掉，所以需要修改配置文件。

打开 Docker Desktop，进入 `Setting` 中的 `Docker Engine`，在 JSON 中插入以下内容（前面的条目要加上逗号）：

```json
"registry-mirrors": [
    "https://docker.1ms.run",
    "https://dockerproxy.net",
    "https://proxy.vvvv.ee",
    "https://dockerproxy.link"
  ]
```

### 2. 仿真环境配置教程

2.1. 下载 `rknn_toolkit2-2.3.2 xxxx.whl` 并将其放在 `rknn_sim_deploy` 目录中，Dockerfile 会运行 pip 指令安装此软件包。

2.2. 通过 `cd` 进入 `rknn_sim_deploy` 目录。

2.3. 构建 Docker 镜像：

```bash
docker build -t rknn-toolkit2:v1 .
```

2.4. 启动并挂载目录，把该目录映射到容器里的 `/workspace` 目录。

```bash

docker run -it --name rknn_sim -v .:/workspace rknn-toolkit2:v1 /bin/bash
cd workspace
```

2.5. 进入环境，尝试以下指令，若无报错，则安装完成：

```bash
python -c "from rknn.api import RKNN; print('RKNN Installed!')"
```

### 3. 使用仿真环境进行 RKNN 模型性能测试

##### 3.1. 导出特定形式的 `ONNX` 模型

在将 YOLOv8 / YOLO11 等现代目标检测模型部署到 RKNN 时，
不能直接使用官方默认的 `yolo export` 命令。
标准的 YOLO 模型会在网络的最末端（Detect 头）包含大量的复杂非线性操作，
例如 DFL 坐标解码、Softmax、Sigmoid 以及特征张量拼接（Concat）。
这些操作对于 INT8 量化极其不友好。如果强行将其打包送入 NPU 计算，
会因超出 INT8 数值表示范围而导致精度瞬间坍塌，
输出结果全为 0.0 或产生异常高得离谱的置信度。

因此，行业内的标准做法是导出一个 “砍头版”（Headless）模型：
即通过代码强行剔除 Detect 层的后处理逻辑，
让模型只输出未经解码的纯粹卷积特征图。后续复杂的坐标映射和 NMS 等后处理，
则统一交由外部的 CPU 使用 Python 或 C++ 来完成。

为了绕过官方库各种版本之间的依赖冲突，
我们在项目目录中提供了一个直接导出脚本 `pt2onnx.py`。
请确保你训练好的 `best.pt` 权重文件与该脚本在同一目录下。

拿到生成的 `best.onnx` 后，强烈建议使用 [Netron](https://netron.app/) 可视化工具打开查看模型尾部结构。
正确的模型最底部不能只有一个 1x...x8400 维度的独立输出节点。它应该呈现出 3 个（或更多）独立并列的输出分支（分别对应大、中、小三种特征图的原始卷积输出）。确认无误后，即可进入下一步的量化环节。

##### 3.2. datasets 校准量化

RKNN 的 `dataset.txt` 用来做模型校准，当你把高精度的 ONNX 模型压缩成低精度的 RKNN（INT8）模型时，NPU 需要拿它来测量数据分布（找最大值和最小值），
以防止精度坍塌，便于将 FLOAT16 格式的数据转化成 INT8。
该文件中写着大概 50 到 200 张图片的路径，格式如下：

```
./calibration_imgs/test_img_1.jpg
./calibration_imgs/test_img_2.jpg
...
```

从原始数据构建校准数据，请参考 `prepare_dataset.py`，只需要将存有图片的 `raw_imgs` 文件夹放置于该目录中，
即可自动生成 `calibration_imgs` 文件夹和 `dataset.txt`。
