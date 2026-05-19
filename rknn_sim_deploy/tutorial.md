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

docker run -it --name rknn_sim -v YOUR:\PROJECT:\rknn_sim_deploy:/workspace rknn-toolkit2:v1 /bin/bash
cd workspace
```

2.5. 进入环境，尝试以下指令，若无报错，则安装完成：

```bash
python -c "from rknn.api import RKNN; print('RKNN Installed!')"
```

### 3. 使用仿真环境进行 RKNN 模型性能测试

##### 3.1. datasets 校准

RKNN 的 `dataset.txt` 用来做模型校准，当你把高精度的 ONNX 模型压缩成低精度的 RKNN（INT8）模型时，NPU 需要拿它来测量数据范围（找最大值和最小值），防止精度坍塌。
该文件中写着大概 20 到 50 张图片的路径，格式如下：

```
./calibration_imgs/test_img_1.jpg
./calibration_imgs/test_img_2.jpg
...
```

在实际工程中，当训练完模型并准备转换成 RK3568 模型时，需要：
1. 打开训练包的 `datasets` 目录下的验证集（`val/images`）。
2. 从里面随机挑出 50 张包含目标的图片。
3. 把这 50 张图片的路径写进 `dataset.txt` 里。
4. 把这个 `dataset.txt` 喂给 RKNN 去做量化校准。

##### 3.2. 