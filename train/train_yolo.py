import os
from ultralytics import YOLO
from datetime import datetime

from config.settings import cfg


def main():
    # 替换为 导出 data.yaml 的实际路径
    local_data_yaml = "./datasets/data.yaml"

    if not os.path.exists(local_data_yaml):
        raise FileNotFoundError(f"[train_yolo:main] 找不到数据集配置文件，请检查路径: {local_data_yaml}")

    print(f"[train:train_yolo:main] 已定位本地数据集: {local_data_yaml}")

    # 边缘部署首选 'n' (Nano) 模型，兼顾极高的运行速度和极低的资源占用
    # 如果想稍微提升精度且设备跑得动，可以改成 "yolo11s.pt"
    edge_weights = "yolo11s.pt"
    print(f"[train:train_yolo:main] 正在初始化预训练模型: {edge_weights}...")
    model = YOLO(edge_weights)
    today = datetime.now().strftime("%Y%m%d")
    print("[train:train_yolo:main] 开始训练...")
    _ = model.train(
        data=local_data_yaml,
        epochs=150,
        patience=30,  # 早停耐心值
        batch=32,
        imgsz=640,  # 输入图片尺寸 (边缘端建议调小)
        device=0,  # GPU 编号 (或 'cpu')
        project="Excavator_Counting_Training",  # 训练结果主目录
        name=f"yolov11s_edge_v1_{today}",  # 本次实验名称
    )

    # 导出 .onnx 格式的模型，根据需要自行调整
    # 命令行指令：yolo export model=best.pt format=onnx imgsz=480 simplify=True...
    # 另注：YOLO imgsz 命令行参数通常是 高,宽，所以导出长方形视频流，应该是 540,960
    print(f"[train:train_yolo:main] 训练完成！正在导出...")
    model.export(
        format="onnx",
        simplify=True,  # 必须精简模型结构，否则 NPU 可能遇到不支持的复杂算子
        half=False,  # NPU 量化需要浮点数作为输入，保持 False
        dynamic=False,  # RKNN 只支持静态图
        imgsz=(384, 640),
    )
    print("[train:train_yolo:main] 边缘模型导出完毕！")


if __name__ == "__main__":
    main()
