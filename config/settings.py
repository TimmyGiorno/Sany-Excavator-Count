import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    # ==========================================
    # 1. 账号与私密配置 (从 .env 动态读取)
    # ==========================================
    ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")
    ROBOFLOW_WORKSPACE_NAME = os.getenv("ROBOFLOW_WORKSPACE_NAME")
    ROBOFLOW_PROJECT_NAME = os.getenv("ROBOFLOW_PROJECT_NAME")

    # 获取版本号并转为整数，如果没有配置默认给 1
    ROBOFLOW_PROJECT_VERSION = int(os.getenv("ROBOFLOW_PROJECT_VERSION", 1))

    # ==========================================
    # 2. YOLO 训练超参数配置 (非敏感信息，直接写在这里)
    # ==========================================
    VERSION_FOR_DOWNLOAD_DATASET = "yolov8"
    MODEL_WEIGHTS = "yolov8n.pt"  # 初始权重
    EPOCHS = 150  # 训练轮数
    PATIENCE = 30  # 早停耐心值
    BATCH_SIZE = 16  # 批次大小
    IMG_SIZE = 416  # 输入图片尺寸 (边缘端建议调小)
    DEVICE = 0  # GPU 编号 (或 'cpu')

    # ==========================================
    # 3. 存储路径配置
    # ==========================================
    TRAIN_PROJECT_DIR = "Excavator_Counting"  # 训练结果主目录
    EXPERIMENT_NAME = "yolov8n_edge_v1"  # 本次实验名称
    EXPORT_FORMAT = "ncnn"  # 边缘端导出格式

cfg = Config()