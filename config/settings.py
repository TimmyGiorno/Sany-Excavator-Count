import os
from dotenv import load_dotenv

load_dotenv()

class Config:

    # Roboflow 账号与私密配置 (从 .env 动态读取)
    ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")
    ROBOFLOW_WORKSPACE_NAME = os.getenv("ROBOFLOW_WORKSPACE_NAME")
    ROBOFLOW_PROJECT_NAME = os.getenv("ROBOFLOW_PROJECT_NAME")

    # 获取版本号并转为整数，如果没有配置默认给 1
    ROBOFLOW_PROJECT_VERSION = int(os.getenv("ROBOFLOW_PROJECT_VERSION", 1))

cfg = Config()
