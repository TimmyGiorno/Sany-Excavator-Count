# Windows RKNN Simulation Environment Deployment Guide Based on Docker

### 0. Relevant Links

*   Tutorial: https://doc.embedfire.com/linux/rk356x/Ai/zh/latest/lubancat_ai/env/toolkit2.html
*   Tutorial: https://www.cnblogs.com/ttkwzyttk/p/19541388
*   RKNN Toolkit Official Repository: https://github.com/airockchip/rknn-toolkit2

### 1. WSL Environment Configuration and Troubleshooting

Note: To use Docker, please check if the latest version of `WSL` is installed on your computer. You can use the following command to check:

```BASH
wsl --verion
```

If the `wsl` version is not output, it means WSL is not installed correctly. Install it using the following commands:

```bash
wsl --install
wsl --update
```

Additionally, run the following command in an Administrator Terminal:

```bash
bcdedit /set hypervisorlaunchtype auto
```

Note: As many domestic Docker image mirrors are no longer available, you need to modify the configuration file.

Open Docker Desktop, go to `Settings` -> `Docker Engine`, and insert the following content into the JSON (remember to add a comma after the previous entry):

```json
"registry-mirrors": [
    "https://docker.1ms.run",
    "https://dockerproxy.net",
    "https://proxy.vvvv.ee",
    "https://dockerproxy.link"
  ]
```

### 2. Simulation Environment Configuration Tutorial

2.1. Download `rknn_toolkit2-2.3.2 xxxx.whl` and place it in the `rknn_sim_deploy` directory. The Dockerfile will run pip to install this package.

2.2. Navigate into the `rknn_sim_deploy` directory via `cd`.

2.3. Build the Docker image:

```bash
docker build -t rknn-toolkit2:v1 .
```

2.4. Start and mount the directory, mapping the current directory to `/workspace` in the container:

```bash
docker run -it --name rknn_sim -v .:/workspace rknn-toolkit2:v1 /bin/bash
cd workspace
```

After entering the directory, you can use the following command to enter the environment subsequently:

```bash
docker exec -it rknn_sim /bin/bash
```

2.5. Enter the environment and run the following command. If there is no error, the installation is successful:

```bash
python -c "from rknn.api import RKNN; print('RKNN Installed!')"
```

### 3. Using the Simulation Environment for RKNN Model Performance Testing

##### 3.1. Exporting the ONNX Model in a Specific Format

When deploying modern object detection models like YOLOv8 / YOLO11 to RKNN,
you cannot directly use the official default `yolo export` command.
Standard YOLO models contain a large number of complex non-linear operations at the very end (the Detect head),
such as DFL coordinate decoding, Softmax, Sigmoid, and feature tensor concatenation (Concat).
These operations are extremely unfriendly to `INT8` quantization. If forced into the NPU for calculation,
precision will collapse instantly due to exceeding the `INT8` numerical range,
resulting in outputs of all `0.0` or abnormally high confidence levels.

Therefore, the industry standard practice is to export a "Headless" model:
Remove the post-processing logic of the Detect layer via code,
letting the model output only the raw convolutional feature maps without decoding. Subsequent complex post-processing such as coordinate mapping and NMS is then unified and completed by an external CPU using Python or C++.

To bypass dependency conflicts between various versions of official libraries,
we provide two export scripts, `pt2onnx_siamese.py` and `pt2onnx_yolo.py`, in the project directory.
Please refer to the code for details.

After obtaining the generated `best.onnx`, it is strongly recommended to use the [Netron](https://netron.app/) visualization tool to open and check the tail structure of the model.
A correct YOLO model should not have a single 1x...x8400 dimension output node at the bottom. Instead, it should present 3 (or more) independent parallel output branches (corresponding to the raw convolutional outputs of large, medium, and small feature maps). After confirmation, proceed to the next step of quantization.

##### 3.2. Dataset Calibration and Quantization

RKNN's `dataset.txt` is used for model calibration. When compressing a high-precision ONNX model into a low-precision RKNN (INT8) model, the NPU needs it to measure the data distribution (find the maximum and minimum values)
to prevent precision collapse when converting FLOAT16 data to INT8.
The file contains paths to approximately 50 to 200 images, formatted as follows:

```text
./calibration_imgs/test_img_1.jpg
./calibration_imgs/test_img_2.jpg
...
```

*   To build a calibration dataset for YOLO from raw data, please refer to `prepare_dataset_yolo.py`. Simply place the folder containing the pictures, `raw_imgs_yolo`, in this directory,
    and it will automatically generate the `calibration_imgs` folder and `dataset_yolo.txt`.
*   For building a calibration dataset for the Siamese model, please refer to the `truck_extract.py` file in the `infer` package.
    `prepare_dataset_yolo.py` is only used to generate `dataset_siamese.txt`.

##### 3.3. Run Model Quantization, Export, and Video Testing

In the Docker environment, run the following command for testing:

```bash
python infer_video_and_export.py
```

The exported model and output results are located in the `tmp_files` directory of this folder.
