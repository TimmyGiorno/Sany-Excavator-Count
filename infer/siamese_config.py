# py
from pathlib import Path


class Config:
    """全局配置"""

    # 数据路径
    annotation_file = r"E:\pycharmProjects\Sany-Excavator-Count\infer\siamese_trainset_V1.1\cvnet_annotations.json"
    image_root = r"E:\pycharmProjects\Sany-Excavator-Count\infer\siamese_trainset_V1.1"

    # 训练参数
    batch_size = 64
    learning_rate = 0.001
    num_epochs = 100
    num_workers = 4

    # 模型参数
    input_size = 224
    embedding_dim = 256
    backbone = 'mobilenet_v3_small'  # 可选: mobilenetv3_small, mobilenetv3_large,
    # resnet18, resnet34, resnet50, resnet101,
    # efficientnet_b0, efficientnet_b1, efficientnet_b2,
    # vit_b_16, vit_b_32, swin_t, swin_s, convnext_tiny

    # 注意力机制参数
    use_spatial_attention = True
    use_channel_attention = True
    attention_position = 'both'  # 'before', 'after', 'both'

    # 多尺度特征参数（新增）
    use_multi_scale = True  # 是否使用多尺度特征提取
    use_local_attention = True  # 是否使用局部区域注意力

    # 损失函数参数
    margin = 0.6

    # 数据划分比例
    train_ratio = 0.8
    val_ratio = 0.1
    test_ratio = 0.1

    # 添加到 config 中
    use_balanced_sampling = True  # 使用平衡采样
    use_hard_mining = True  # 使用难负样本挖掘

    hard_negative_threshold = 0.7  # 超过此相似度的负样本视为难负样本
    hard_mining_interval = 5  # 每N个epoch挖掘一次
    hard_negative_weight = 5.0  # 难负样本的采样权重倍数

    # 添加ViT相关配置
    model_type = 'lightweight_vit'  # 'vit', 'lightweight_vit', 'enhanced_siamese'

    # ViT特定参数
    vit_patch_size = 16
    vit_embed_dim = 256  # 轻量级用256，完整版用384
    vit_num_heads = 4
    vit_num_layers = 6
    vit_dropout = 0.1
    vit_use_cross_attention = True
    vit_aggregation = 'cls'  # 'cls', 'mean', 'attention'
    vit_learning_rate = 1e-4  # ViT通常需要更小的学习率
    vit_weight_decay = 0.05

    # 随机种子
    random_seed = 42

    # 设备
    device = 'cuda' if __import__('torch').cuda.is_available() else 'cpu'

    # 保存路径
    save_dir = Path(r"E:\pycharmProjects\Sany-Excavator-Count\infer\runs\siamese_model_V1.1")

    @classmethod
    def to_dict(cls):
        return {k: str(v) if isinstance(v, Path) else v
                for k, v in cls.__dict__.items()
                if not k.startswith('_') and not callable(v)}


# 创建配置实例
config = Config()
