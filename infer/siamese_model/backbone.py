# models/backbone.py
import torch.nn as nn
from torchvision import models


class BackboneFactory:
    """主干网络工厂类 - 支持多种backbone"""

    _registry = {}

    @classmethod
    def register(cls, name, builder):
        cls._registry[name] = builder

    @classmethod
    def create(cls, name, pretrained=True):
        if name not in cls._registry:
            raise ValueError(f"不支持的backbone: {name}. 支持: {list(cls._registry.keys())}")
        return cls._registry[name](pretrained)


def _create_mobilenetv3_small(pretrained):
    base = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None)
    return nn.Sequential(*list(base.children())[:-2]), 576


def _create_mobilenetv3_large(pretrained):
    base = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pretrained else None)
    return nn.Sequential(*list(base.children())[:-2]), 960


def _create_resnet18(pretrained):
    base = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    return nn.Sequential(*list(base.children())[:-2]), 512


def _create_resnet34(pretrained):
    base = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None)
    return nn.Sequential(*list(base.children())[:-2]), 512


def _create_resnet50(pretrained):
    base = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
    return nn.Sequential(*list(base.children())[:-2]), 2048


def _create_resnet101(pretrained):
    base = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1 if pretrained else None)
    return nn.Sequential(*list(base.children())[:-2]), 2048


def _create_efficientnet_b0(pretrained):
    base = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
    return nn.Sequential(*list(base.children())[:-2]), 1280


def _create_efficientnet_b1(pretrained):
    base = models.efficientnet_b1(weights=models.EfficientNet_B1_Weights.IMAGENET1K_V1 if pretrained else None)
    return nn.Sequential(*list(base.children())[:-2]), 1280


def _create_efficientnet_b2(pretrained):
    base = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.IMAGENET1K_V1 if pretrained else None)
    return nn.Sequential(*list(base.children())[:-2]), 1408


def _create_vit_b_16(pretrained):
    base = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None)
    # ViT输出是 [B, 768]，需要特殊处理
    return base, 768, True  # 返回是否是ViT标志


def _create_vit_b_32(pretrained):
    base = models.vit_b_32(weights=models.ViT_B_32_Weights.IMAGENET1K_V1 if pretrained else None)
    return base, 768, True


def _create_swin_t(pretrained):
    base = models.swin_t(weights=models.Swin_T_Weights.IMAGENET1K_V1 if pretrained else None)
    return base, 768, True


def _create_swin_s(pretrained):
    base = models.swin_s(weights=models.Swin_S_Weights.IMAGENET1K_V1 if pretrained else None)
    return base, 768, True


def _create_convnext_tiny(pretrained):
    base = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None)
    return nn.Sequential(*list(base.children())[:-2]), 768


# 注册所有backbone
BackboneFactory.register('mobilenetv3_small', _create_mobilenetv3_small)
BackboneFactory.register('mobilenetv3_large', _create_mobilenetv3_large)
BackboneFactory.register('resnet18', _create_resnet18)
BackboneFactory.register('resnet34', _create_resnet34)
BackboneFactory.register('resnet50', _create_resnet50)
BackboneFactory.register('resnet101', _create_resnet101)
BackboneFactory.register('efficientnet_b0', _create_efficientnet_b0)
BackboneFactory.register('efficientnet_b1', _create_efficientnet_b1)
BackboneFactory.register('efficientnet_b2', _create_efficientnet_b2)
BackboneFactory.register('vit_b_16', _create_vit_b_16)
BackboneFactory.register('vit_b_32', _create_vit_b_32)
BackboneFactory.register('swin_t', _create_swin_t)
BackboneFactory.register('swin_s', _create_swin_s)
BackboneFactory.register('convnext_tiny', _create_convnext_tiny)