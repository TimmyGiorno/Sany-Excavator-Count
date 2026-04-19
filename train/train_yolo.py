import os
from ultralytics import YOLO

from config.settings import cfg


def main():
    # ==========================================
    # 1. 配置本地数据集路径
    # ==========================================
    local_data_yaml = "datasets/data.yaml"  # 替换为你的 CVAT 导出 data.yaml.yaml 的实际路径

    if not os.path.exists(local_data_yaml):
        raise FileNotFoundError(f"找不到数据集配置文件，请检查路径: {local_data_yaml}")

    print(f">>> 已定位本地数据集: {local_data_yaml}")

    # ==========================================
    # 2. 初始化适合边缘部署的轻量化模型
    # ==========================================
    # 边缘部署首选 'n' (Nano) 模型，兼顾极高的运行速度和极低的资源占用
    edge_weights = "yolo11s.pt"  # 如果你想稍微提升精度且设备跑得动，可以改成 "yolo11s.pt"
    print(f">>> 正在初始化边缘部署专供模型: {edge_weights}...")
    model = YOLO(edge_weights)

    # ==========================================
    # 3. 开始模型训练
    # ==========================================
    print(">>> 开始训练...")
    _ = model.train(
        data=local_data_yaml,
        epochs=cfg.EPOCHS,
        patience=cfg.PATIENCE,
        batch=cfg.BATCH_SIZE,
        imgsz=cfg.IMG_SIZE,
        device=cfg.DEVICE,
        project=cfg.TRAIN_PROJECT_DIR,
        name=cfg.EXPERIMENT_NAME,
    )

    # ==========================================
    # 4. 导出模型 (针对边缘端的特殊提醒)
    # ==========================================
    print(f"\n>>> 训练完成！正在导出 {cfg.EXPORT_FORMAT} 格式...")
    # half=True (FP16半精度) 非常适合边缘端，能提升速度并减少内存占用
    model.export(format=cfg.EXPORT_FORMAT, half=False)
    print(">>> 导出完毕！")


if __name__ == "__main__":
    main()