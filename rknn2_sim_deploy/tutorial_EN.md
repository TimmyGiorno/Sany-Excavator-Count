# Operational Guide for Deploying Windows RKNN Simulation Environment via Docker

### 0. Related Links

* Related Tutorial: https://doc.embedfire.com/linux/rk356x/Ai/zh/latest/lubancat_ai/env/toolkit2.html
* Related Tutorial: https://www.cnblogs.com/ttkwzyttk/p/19541388
* Official RKNN Toolkit Repository: https://github.com/airockchip/rknn-toolkit2

### 1. WSL Environment Configuration and Troubleshooting

Note: To use Docker, please verify whether the latest version of `WSL` is installed on your computer. You can check this using the following command:

```bash
wsl --version
```

If the wsl version is not output, it means wsl is not installed correctly. Use the following commands to install it:

```bash
wsl --install
wsl --update
```

Then, run the following command in an Administrator Terminal:

```Bash
bcdedit /set hypervisorlaunchtype auto
```

Additional Note: Since many Docker image mirrors in mainland China are currently down, you need to modify the configuration file.

Open Docker Desktop, go to Docker Engine under Settings, and insert the following content into the JSON configuration (remember to add a comma to the preceding entry):

```JSON
"registry-mirrors": [
"[https://docker.1ms.run](https://docker.1ms.run)",
"[https://dockerproxy.net](https://dockerproxy.net)",
"[https://proxy.vvvv.ee](https://proxy.vvvv.ee)",
"[https://dockerproxy.link](https://dockerproxy.link)"
]
```

### 2. Simulation Environment Configuration Guide

2.1. Download rknn_toolkit2-2.3.2 xxxx.whl and place it in the rknn_sim_deploy directory. The Dockerfile will execute a pip command to install this package.

2.2. Enter the rknn_sim_deploy directory using the cd command.

2.3. Build the Docker image:

```Bash
docker build -t rknn-toolkit2:v1 .
```

2.4. Start the container and mount the directory, mapping it to the /workspace directory inside the container.

```bash
docker run -it --name rknn_sim -v .:/workspace rknn-toolkit2:v1 /bin/bash
cd workspace
```

2.5. Once inside the environment, try the following command. If no errors occur, the installation is complete:

```bash
python -c "from rknn.api import RKNN; print('RKNN Installed!')"
```

### 3. Conducting RKNN Model Performance Testing Using the Simulation Environment

##### 3.1. Exporting a Specific Format of the ONNX Model

When deploying modern object detection models like YOLOv8 / YOLO11 to RKNN, you cannot directly use the official default yolo export command. Standard YOLO models contain a large number of complex non-linear operations at the very end of the network (the Detect head), such as DFL coordinate decoding, Softmax, Sigmoid, and feature tensor concatenation (Concat). These operations are extremely unfriendly to INT8 quantization. If forcibly packaged and sent to the NPU for computation, the precision will instantly collapse because the values exceed the INT8 representation range, resulting in outputs that are all 0.0 or abnormally high confidence scores.

Therefore, the standard industry practice is to export a _Headless_ model: this means forcibly removing the post-processing logic of the Detect layer via code, allowing the model to output only pure, undecoded convolutional feature maps. Subsequent complex operations like coordinate mapping and NMS post-processing are then handled uniformly by the external CPU using Python or C++.

To bypass the dependency conflicts between various versions of the official libraries, we have provided a direct export script `pt2onnx.py` in the project directory. Please ensure that your trained best.pt weight file is in the same directory as this script.

After obtaining the generated `best.onnx`, it is highly recommended to open and inspect the tail structure of the model using the [Netron visualization tool](https://netron.app/). A correct model should not have only a single independent output node with a dimension like 1x...x8400 at the very bottom. Instead, it should present 3 (or more) independent parallel output branches (corresponding to the raw convolutional outputs for large, medium, and small feature maps, respectively). Once confirmed, you can proceed to the next quantization step.

##### 3.2. Datasets Calibration and Quantization
The `dataset.txt` file in RKNN is used for model calibration. When compressing a high-precision ONNX model into a low-precision RKNN (INT8) model, the NPU needs it to measure the data distribution (finding the maximum and minimum values) to prevent precision collapse and facilitate the conversion of data from FLOAT16 format to INT8. This file contains the paths of approximately 50 to 200 images, formatted as follows:

```angular2html
./calibration_imgs/test_img_1.jpg
./calibration_imgs/test_img_2.jpg
...
```

To build calibration data from the raw data, please refer to `prepare_dataset.py`. Simply place the raw_imgs folder containing the images into this directory, and it will automatically generate the `calibration_imgs` folder and the dataset.txt file.