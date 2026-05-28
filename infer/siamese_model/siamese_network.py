# models/siamese_network.py
import torch
import torch.nn as nn
from .backbone import BackboneFactory
from .attention import CBAM


class AttentionSiameseNetwork(nn.Module):
    """
    带注意力机制的轻量孪生网络
    支持多种backbone
    """

    def __init__(self, embedding_dim=256, backbone='mobilenetv3_small',
                 use_spatial_attention=True, use_channel_attention=True,
                 attention_position='both', pretrained=True):
        super(AttentionSiameseNetwork, self).__init__()

        self.embedding_dim = embedding_dim
        self.use_spatial_attention = use_spatial_attention
        self.use_channel_attention = use_channel_attention
        self.attention_position = attention_position
        self.backbone_name = backbone

        # 创建主干网络
        backbone_result = BackboneFactory.create(backbone, pretrained)

        if len(backbone_result) == 3:
            self.backbone, feature_dim, self.is_vit = backbone_result
        else:
            self.backbone, feature_dim = backbone_result
            self.is_vit = False

        # 注意力模块
        if use_channel_attention or use_spatial_attention:
            self.attention = CBAM(feature_dim, reduction=16, kernel_size=7)
        else:
            self.attention = nn.Identity()

        # 全局池化（ViT不需要池化）
        if not self.is_vit:
            self.global_pool = nn.AdaptiveAvgPool2d(1)

        # 投影头
        self.projection = nn.Sequential(
            nn.Linear(feature_dim, embedding_dim * 2),
            nn.BatchNorm1d(embedding_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.BatchNorm1d(embedding_dim)
        )

    def forward_single(self, x):
        """提取单张图片的特征向量"""
        features = self.backbone(x)

        if self.is_vit:
            # ViT输出已经是 [B, feat_dim]
            embedding = self.projection(features)
        else:
            if self.attention_position == 'before':
                features = self.attention(features)

            features = self.global_pool(features)
            features = features.flatten(1)
            embedding = self.projection(features)

        embedding = nn.functional.normalize(embedding, p=2, dim=1)
        return embedding

    def forward(self, img1, img2):
        """孪生网络前向传播"""
        embedding1 = self.forward_single(img1)
        embedding2 = self.forward_single(img2)
        similarity = (embedding1 * embedding2).sum(dim=1)
        return similarity, embedding1, embedding2

    def get_embedding(self, img):
        """获取单张图片的特征向量（用于检索）"""
        return self.forward_single(img)

    def compare(self, img1, img2):
        """比较两张图片，返回相似度"""
        with torch.no_grad():
            similarity, _, _ = self.forward(img1, img2)
        return similarity


# 模型加载函数
def load_siamese_model(model_path, device='cuda', **kwargs):
    """加载训练好的模型"""
    checkpoint = torch.load(model_path, map_location=device)

    model = AttentionSiameseNetwork(
        embedding_dim=checkpoint.get('embedding_dim', kwargs.get('embedding_dim', 256)),
        backbone=checkpoint.get('config', {}).get('backbone', kwargs.get('backbone', 'mobilenetv3_small')),
        use_spatial_attention=checkpoint.get('config', {}).get('use_spatial_attention', True),
        use_channel_attention=checkpoint.get('config', {}).get('use_channel_attention', True),
        attention_position=checkpoint.get('config', {}).get('attention_position', 'both'),
        pretrained=False
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    return model, checkpoint