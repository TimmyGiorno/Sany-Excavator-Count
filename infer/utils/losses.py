# utils/losses.py
import torch
import torch.nn as nn


class ContrastiveLoss(nn.Module):
    """对比损失"""
    def __init__(self, margin=0.5):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, similarity, labels):
        similarity = (similarity + 1) / 2
        pos_loss = (1 - similarity) * labels
        neg_loss = similarity * (1 - labels)
        neg_loss = torch.clamp(neg_loss - self.margin, min=0)
        return (pos_loss + neg_loss).mean()


class FocalLoss(nn.Module):
    """Focal Loss"""
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        bce_loss = - (targets * torch.log(inputs + 1e-7) +
                      (1 - targets) * torch.log(1 - inputs + 1e-7))
        pt = torch.where(targets == 1, inputs, 1 - inputs)
        alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        return (alpha_t * (1 - pt) ** self.gamma * bce_loss).mean()