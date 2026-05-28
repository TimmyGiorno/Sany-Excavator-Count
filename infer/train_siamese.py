# train.py
import os
import random
import json
from pathlib import Path
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from siamese_config import config
from siamese_model.siamese_network import AttentionSiameseNetwork
from utils.dataset import TruckPairDataset, load_annotations
from utils.transforms import get_transforms
from utils.losses import ContrastiveLoss
from utils.metrics import compute_metrics


def split_dataset(annotations, train_ratio, val_ratio, test_ratio, random_seed):
    """划分数据集"""
    random.seed(random_seed)
    torch.manual_seed(random_seed)

    shuffled = annotations.copy()
    random.shuffle(shuffled)

    total = len(shuffled)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    return (shuffled[:train_end],
            shuffled[train_end:val_end],
            shuffled[val_end:])


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []

    pbar = tqdm(loader, desc='Training', unit='batch', ncols=100)
    for img1, img2, labels in pbar:
        img1, img2, labels = img1.to(device), img2.to(device), labels.float().to(device)

        similarity, _, _ = model(img1, img2)
        loss = criterion(similarity, labels)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        preds = (similarity > 0.5).float()
        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())

        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    metrics = compute_metrics(all_labels, all_preds)
    return total_loss / len(loader), metrics


def validate_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds, all_labels, all_probs = [], [], []

    pbar = tqdm(loader, desc='Validation', unit='batch', ncols=100)
    with torch.no_grad():
        for img1, img2, labels in pbar:
            img1, img2, labels = img1.to(device), img2.to(device), labels.float().to(device)

            similarity, _, _ = model(img1, img2)
            loss = criterion(similarity, labels)

            total_loss += loss.item()
            preds = (similarity > 0.5).float()
            all_preds.extend(preds.detach().cpu().numpy())
            all_labels.extend(labels.detach().cpu().numpy())
            all_probs.extend(similarity.detach().cpu().numpy())

            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    metrics = compute_metrics(all_labels, all_preds, all_probs)
    return total_loss / len(loader), metrics


def save_model(model, save_path, epoch, optimizer, best_val_f1):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_f1': best_val_f1,
        'embedding_dim': model.embedding_dim,
        'config': {
            'backbone': model.backbone_name,
            'use_spatial_attention': model.use_spatial_attention,
            'use_channel_attention': model.use_channel_attention,
            'attention_position': model.attention_position,
        }
    }, save_path)
    print(f"模型已保存: {save_path}")


def main():
    print("=" * 60)
    print("带注意力机制的轻量孪生网络训练脚本")
    print("=" * 60)
    print(f"设备: {config.device}")
    print(f"主干网络: {config.backbone}")
    print(f"特征维度: {config.embedding_dim}")

    # 1. 加载数据
    annotations = load_annotations(config.annotation_file)
    print(f"加载了 {len(annotations)} 个标注样本")

    # 2. 划分数据集
    train_anns, val_anns, test_anns = split_dataset(
        annotations, config.train_ratio, config.val_ratio, config.test_ratio, config.random_seed
    )

    # 3. 创建数据集
    train_transform = get_transforms(config.input_size, is_train=True)
    val_transform = get_transforms(config.input_size, is_train=False)

    train_dataset = TruckPairDataset(train_anns, config.image_root, train_transform)
    val_dataset = TruckPairDataset(val_anns, config.image_root, val_transform)
    test_dataset = TruckPairDataset(test_anns, config.image_root, val_transform)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

    # 4. 创建模型
    model = AttentionSiameseNetwork(
        embedding_dim=config.embedding_dim,
        backbone=config.backbone,
        use_spatial_attention=config.use_spatial_attention,
        use_channel_attention=config.use_channel_attention,
        attention_position=config.attention_position
    )
    model = model.to(config.device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"总参数: {total_params:,}")

    # 5. 训练
    criterion = ContrastiveLoss(margin=config.margin)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.num_epochs, eta_min=1e-6)

    best_val_f1 = 0
    patience = 15
    no_improve = 0

    for epoch in range(config.num_epochs):
        train_loss, train_metrics = train_epoch(model, train_loader, criterion, optimizer, config.device)
        val_loss, val_metrics = validate_epoch(model, val_loader, criterion, config.device)
        scheduler.step()

        print(f"\nEpoch {epoch + 1}/{config.num_epochs} | LR: {optimizer.param_groups[0]['lr']:.6f}")
        print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_metrics['accuracy']:.4f}")
        print(f"  Val   - Loss: {val_loss:.4f}, F1: {val_metrics['f1']:.4f}, AUC: {val_metrics['auc']:.4f}")

        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            no_improve = 0
            save_path = config.save_dir / 'attention_siamese_best.pth'
            save_model(model, save_path, epoch, optimizer, best_val_f1)
            print(f"  *** 新最佳模型! (F1: {best_val_f1:.4f}) ***")
        else:
            no_improve += 1
            if no_improve >= patience and epoch >= 10:
                print(f"\n早停触发！最佳F1: {best_val_f1:.4f}")
                break

    print("\n训练完成!")


if __name__ == "__main__":
    main()