# config.py
from pathlib import Path


class Config:
    """全局配置"""

    # 数据路径
    annotation_file = r"E:\pycharmProjects\Sany-Excavator-Count\infer\cvnet_annotations.json"
    image_root = r"E:\pycharmProjects\Sany-Excavator-Count\infer\cvnet_trainset"

    # 训练参数
    batch_size = 64
    learning_rate = 0.001
    num_epochs = 100
    num_workers = 4

    # 模型参数
    input_size = 224
    embedding_dim = 256
    backbone = 'mobilenetv3_small'  # 可选: mobilenetv3_small, mobilenetv3_large,
    # resnet18, resnet34, resnet50, resnet101,
    # efficientnet_b0, efficientnet_b1, efficientnet_b2,
    # vit_b_16, vit_b_32, swin_t, swin_s, convnext_tiny

    # 注意力机制参数
    use_spatial_attention = True
    use_channel_attention = True
    attention_position = 'both'  # 'before', 'after', 'both'

    # 损失函数参数
    margin = 0.5

    # 数据划分比例
    train_ratio = 0.8
    val_ratio = 0.1
    test_ratio = 0.1

    # 随机种子
    random_seed = 42

    # 设备
    device = 'cuda' if __import__('torch').cuda.is_available() else 'cpu'

    # 保存路径
    save_dir = Path(r"E:\pycharmProjects\Sany-Excavator-Count\infer\runs\siamese_model")

    @classmethod
    def to_dict(cls):
        return {k: str(v) if isinstance(v, Path) else v
                for k, v in cls.__dict__.items()
                if not k.startswith('_') and not callable(v)}


# 创建配置实例
config = Config()