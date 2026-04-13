from roboflow import Roboflow
from ultralytics import YOLO

from config.settings import cfg


def main():
    # ==========================================
    # 1. 配置 Roboflow
    # ==========================================
    print(">>> 正在从 Roboflow 拉取数据集...")
    rf = Roboflow(api_key=cfg.ROBOFLOW_API_KEY)

    project = rf.workspace(cfg.ROBOFLOW_WORKSPACE_NAME).project(cfg.ROBOFLOW_PROJECT_NAME)
    version = project.version(cfg.ROBOFLOW_PROJECT_VERSION)

    dataset = version.download(cfg.VERSION_FOR_DOWNLOAD_DATASET)
    print(f">>> 数据集已下载至: {dataset.location}")

    # ==========================================
    # 2. 初始化 YOLO 模型
    # ==========================================
    print(f">>> 正在初始化 YOLO 模型: {cfg.MODEL_WEIGHTS}...")
    model = YOLO(cfg.MODEL_WEIGHTS)

    # ==========================================
    # 3. 开始模型训练，可以添加更多参数
    # ==========================================
    print(">>> 开始训练...")
    _ = model.train(
        data=f"{dataset.location}/data.yaml",
        epochs=cfg.EPOCHS,
        patience=cfg.PATIENCE,
        batch=cfg.BATCH_SIZE,
        imgsz=cfg.IMG_SIZE,
        device=cfg.DEVICE,
        project=cfg.TRAIN_PROJECT_DIR,
        name=cfg.EXPERIMENT_NAME,
    )

    # ==========================================
    # 4. 导出模型
    # ==========================================
    print(f"\n>>> 训练完成！正在导出 {cfg.EXPORT_FORMAT} 格式...")
    model.export(format=cfg.EXPORT_FORMAT, half=True)
    print(">>> 导出完毕！")


if __name__ == "__main__":
    main()