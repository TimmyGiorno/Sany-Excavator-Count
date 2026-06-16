import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class LocalCorrelation(nn.Module):
    """
    局部相关性计算模块
    计算 F1 中每个位置与 F2 中邻域位置的相似度
    """

    def __init__(self, patch_size=3, stride=1, neighbor_range=5):
        """
        Args:
            patch_size: 局部patch的大小
            stride: patch的步长
            neighbor_range: 在F2中搜索的邻域半径
        """
        super().__init__()
        self.patch_size = patch_size
        self.stride = stride
        self.neighbor_range = neighbor_range

    def forward(self, f1, f2):
        """
        Args:
            f1: [B, C, H, W] 特征图1
            f2: [B, C, H, W] 特征图2
        Returns:
            corr_map: [B, (2*neighbor_range+1)^2, H', W'] 相关性图
        """
        B, C, H, W = f1.shape

        # 1. 从 f2 中提取所有待搜索的局部patch
        # 使用 unfold 提取滑动窗口
        patches_f2 = F.unfold(
            f2,
            kernel_size=self.patch_size,
            stride=self.stride,
            padding=self.patch_size // 2
        )  # [B, C*P*P, L] 其中 L = H' * W'

        # 重塑: [B, C, P, P, L]
        P = self.patch_size
        patches_f2 = patches_f2.view(B, C, P, P, -1)
        # 再重塑为 [B, L, C, P, P]
        patches_f2 = patches_f2.permute(0, 4, 1, 2, 3)

        # 2. 从 f1 中提取所有锚点patch
        patches_f1 = F.unfold(
            f1,
            kernel_size=self.patch_size,
            stride=self.stride,
            padding=self.patch_size // 2
        )  # [B, C*P*P, L]
        patches_f1 = patches_f1.view(B, C, P, P, -1)
        patches_f1 = patches_f1.permute(0, 4, 1, 2, 3)

        L = patches_f1.shape[1]  # 位置数量
        H_prime = (H + self.stride - 1) // self.stride
        W_prime = (W + self.stride - 1) // self.stride

        # 3. 对于每个位置，只与相邻区域计算相关性
        corr_list = []
        for h in range(H_prime):
            for w in range(W_prime):
                idx = h * W_prime + w
                patch1 = patches_f1[:, idx:idx + 1]  # [B, 1, C, P, P]

                # 确定搜索范围
                h_start = max(0, h - self.neighbor_range)
                h_end = min(H_prime, h + self.neighbor_range + 1)
                w_start = max(0, w - self.neighbor_range)
                w_end = min(W_prime, w + self.neighbor_range + 1)

                # 收集邻域位置的特征
                neighbor_indices = []
                for nh in range(h_start, h_end):
                    for nw in range(w_start, w_end):
                        neighbor_indices.append(nh * W_prime + nw)

                neighbor_patches = patches_f2[:, neighbor_indices]  # [B, K, C, P, P]

                # 计算余弦相似度（比L2距离更适合高维特征）
                patch1_norm = F.normalize(patch1.view(B, 1, -1), dim=2)
                neighbor_norm = F.normalize(neighbor_patches.view(B, -1, C * P * P), dim=2)

                # [B, 1, C*P*P] @ [B, C*P*P, K] = [B, 1, K]
                sim = torch.bmm(patch1_norm, neighbor_norm.transpose(1, 2))
                corr_list.append(sim.squeeze(1))  # [B, K]

        # 重塑为空间特征图
        # 这一步较为复杂，简化实现：直接对所有位置计算全局相关
        # 更高效的做法是使用相关性层(Correlation Layer)，类似FlowNet
        return self.global_correlation(patches_f1, patches_f2, H_prime, W_prime)

    def global_correlation(self, patches_f1, patches_f2, H, W):
        """简化的全局相关计算（作为备选）"""
        B, L, C, P, P = patches_f1.shape

        # 重塑为 [B, L, C*P*P]
        patches_f1_flat = patches_f1.view(B, L, -1)
        patches_f2_flat = patches_f2.view(B, L, -1)

        # 归一化
        patches_f1_norm = F.normalize(patches_f1_flat, dim=2)
        patches_f2_norm = F.normalize(patches_f2_flat, dim=2)

        # 计算相关矩阵 [B, L, L]
        corr_matrix = torch.bmm(patches_f1_norm, patches_f2_norm.transpose(1, 2))

        # 重塑为 [B, L, H, W]，保持空间结构
        corr_map = corr_matrix.view(B, L, H, W)

        return corr_map


class ChannelAttention(nn.Module):
    """通道注意力模块，增强重要特征通道"""

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class SpatialAttention(nn.Module):
    """空间注意力模块，关注重要区域"""

    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avg_out, max_out], dim=1)
        y = self.conv(y)
        return x * self.sigmoid(y)


class EnhancedSiameseNetwork(nn.Module):
    """
    增强版孪生网络：
    - 保留空间信息，不压缩成向量
    - 使用局部相关性计算进行密集对比
    - 加粗的"汇报通道"
    """

    def __init__(self, backbone='resnet34', pretrained=True, use_attention=True):
        super().__init__()

        # 使用较浅的backbone，保留更高分辨率的特征图
        if backbone == 'resnet34':
            import torchvision.models as models
            self.backbone = models.resnet34(weights='DEFAULT' if pretrained else None)
            # 去掉最后的全局池化和全连接层
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])
            self.out_channels = 512
        elif backbone == 'resnet18':
            import torchvision.models as models
            self.backbone = models.resnet18(weights='DEFAULT' if pretrained else None)
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])
            self.out_channels = 512
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # 注意力模块（可选）
        self.use_attention = use_attention
        if use_attention:
            self.channel_att = ChannelAttention(self.out_channels)
            self.spatial_att = SpatialAttention()

        # 局部相关性计算
        self.correlation = LocalCorrelation(patch_size=3, stride=2, neighbor_range=7)

        # 特征聚合头：将相关性图聚合为全局特征
        # 相关性图尺寸：[B, L, H, W]，其中 L = H * W
        self.aggregator = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),  # 这里256是H'*W'，需要调整
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        # 为了处理变长的相关性图，使用自适应池化
        self.adaptive_pool = nn.AdaptiveMaxPool2d((16, 16))

    def forward(self, img1, img2):
        """
        Args:
            img1: [B, 3, H, W]
            img2: [B, 3, H, W]
        Returns:
            similarity: [B, 1] 相似度分数
            feat1: 特征图1（用于可视化）
            feat2: 特征图2（用于可视化）
        """
        # 1. 特征提取
        feat1 = self.backbone(img1)
        feat2 = self.backbone(img2)

        # 2. 注意力增强（可选）
        if self.use_attention:
            feat1 = self.channel_att(feat1)
            feat1 = self.spatial_att(feat1)
            feat2 = self.channel_att(feat2)
            feat2 = self.spatial_att(feat2)

        # 3. 局部相关性计算
        # 获取特征图的空间尺寸
        B, C, H, W = feat1.shape

        # 简化：使用全局平均池化+展平，然后计算逐位置相似度
        # 这是一种更稳定但信息损失稍大的方法

        # 方案：计算两个特征图之间的逐位置余弦相似度
        feat1_flat = feat1.view(B, C, -1)  # [B, C, H*W]
        feat2_flat = feat2.view(B, C, -1)

        # 归一化
        feat1_norm = F.normalize(feat1_flat, dim=1)
        feat2_norm = F.normalize(feat2_flat, dim=1)

        # 计算相似度矩阵 [B, H*W, H*W]
        similarity_matrix = torch.bmm(feat1_norm.transpose(1, 2), feat2_norm)

        # 对每个位置，取最大相似度作为该位置的匹配程度
        max_sim_per_position, _ = similarity_matrix.max(dim=2)  # [B, H*W]

        # 重塑为特征图 [B, 1, H, W]
        correlation_map = max_sim_per_position.view(B, 1, H, W)

        # 4. 使用可变形卷积进一步聚合（可选）
        # 这里简化：直接使用卷积聚合
        agg_feat = self.adaptive_pool(correlation_map)  # [B, 1, 16, 16]

        # 5. 分类头
        similarity = self.aggregator(agg_feat.expand(-1, 128, -1, -1))  # 扩展通道数

        return similarity, feat1, feat2


# 更轻量级的版本：直接拼接特征图 + 深度可分离卷积
class LightweightSiameseNetwork(nn.Module):
    """
    轻量级版本：使用深度可分离卷积处理拼接后的特征图
    相比上面的EnhancedSiameseNetwork，这个更简单、更稳定
    """

    def __init__(self, backbone='mobilenet_v3_small', pretrained=True, use_sigmoid=False):
        super().__init__()
        self.use_sigmoid = use_sigmoid

        if backbone == 'mobilenet_v3_small':
            import torchvision.models as models
            self.backbone = models.mobilenet_v3_small(weights='DEFAULT' if pretrained else None)
            # 去掉分类头，保留到features
            self.backbone = self.backbone.features
            self.out_channels = 576  # MobileNetV3 small的输出通道
        elif backbone == 'mobilenet_v3_large':
            import torchvision.models as models
            self.backbone = models.mobilenet_v3_large(weights='DEFAULT' if pretrained else None)
            self.backbone = self.backbone.features
            self.out_channels = 960
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # 1x1卷积降低通道数
        self.reduce_dim = nn.Conv2d(self.out_channels * 2, 256, kernel_size=1)

        # 深度可分离卷积块，处理拼接后的双路特征
        self.depthwise_sep = nn.Sequential(
            # Depthwise
            nn.Conv2d(256, 256, kernel_size=3, padding=1, groups=256),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # Pointwise
            nn.Conv2d(256, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # 再一层
            nn.Conv2d(128, 128, kernel_size=3, padding=1, groups=128),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # 全局聚合
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        if use_sigmoid:
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(64, 32),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
                nn.Linear(32, 1),
                nn.Sigmoid()
            )
        else:
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(64, 32),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
                nn.Linear(32, 1),
                # 不加 Sigmoid，输出 logits
            )
    def forward(self, img1, img2):
        # 提取特征
        feat1 = self.backbone(img1)
        feat2 = self.backbone(img2)

        # 在通道维度拼接
        concat_feat = torch.cat([feat1, feat2], dim=1)  # [B, 2C, H, W]

        # 降维
        concat_feat = self.reduce_dim(concat_feat)

        # 深度可分离卷积处理
        processed = self.depthwise_sep(concat_feat)

        # 全局聚合
        pooled = self.global_pool(processed)

        # 分类
        similarity = self.classifier(pooled)

        return similarity, feat1, feat2