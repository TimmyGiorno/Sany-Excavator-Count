# train_siamese.py
import os
import random
import json
from pathlib import Path
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torch.nn import BCEWithLogitsLoss
from tqdm import tqdm
import numpy as np
from collections import Counter

from siamese_config import config
from siamese_model.siamese_network import LightweightSiameseNetwork, EnhancedSiameseNetwork
from siamese_model.vit_siamese import ViTSiameseNetwork, LightweightViTSiamese
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


def mine_hard_negatives(model, dataset, device, hard_threshold=0.7, top_k=50):
    """
    挖掘难负样本：找出相似度高的负样本对

    Returns:
        hard_neg_indices: 难负样本在数据集中的索引列表
    """
    model.eval()
    hard_neg_indices = []

    print(f"  挖掘难负样本 (阈值>{hard_threshold})...")

    with torch.no_grad():
        for idx in tqdm(range(len(dataset)), desc="    扫描"):
            img1, img2, label = dataset[idx]

            if label == 0:  # 只关注负样本
                img1_t = img1.unsqueeze(0).to(device)
                img2_t = img2.unsqueeze(0).to(device)

                # 直接调用模型，取第一个返回值（相似度/logits）
                similarity, _, _ = model(img1_t, img2_t)

                # 转换为概率
                if hasattr(model, 'output_is_logits') and model.output_is_logits:
                    sim_value = torch.sigmoid(similarity).item()
                else:
                    sim_value = similarity.item()

                if sim_value > hard_threshold:
                    hard_neg_indices.append((idx, sim_value))

    # 按相似度排序，取 top_k
    hard_neg_indices.sort(key=lambda x: x[1], reverse=True)
    hard_neg_indices = hard_neg_indices[:top_k]

    print(f"    发现 {len(hard_neg_indices)} 个难负样本")
    return [idx for idx, _ in hard_neg_indices]


def create_weighted_sampler(dataset, hard_neg_indices, hard_weight=5.0):
    """
    创建加权采样器，让难负样本被采样的概率更高
    """
    weights = torch.ones(len(dataset))

    # 难负样本权重加倍
    for idx in hard_neg_indices:
        weights[idx] *= hard_weight

    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
    return sampler


def train_epoch(model, loader, criterion, optimizer, device, output_is_logits=True):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []

    pbar = tqdm(loader, desc='Training', unit='batch', ncols=100)
    for img1, img2, labels in pbar:
        img1, img2, labels = img1.to(device), img2.to(device), labels.float().to(device)

        # 前向传播
        outputs, _, _ = model(img1, img2)

        # 计算损失
        if output_is_logits:
            # 模型输出是 logits
            loss = criterion(outputs, labels.unsqueeze(1))
            preds = (torch.sigmoid(outputs) > 0.5).float()
        else:
            # 模型输出已经是概率值（0-1之间）
            loss = criterion(outputs, labels)
            preds = (outputs > 0.5).float()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())

        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    metrics = compute_metrics(all_labels, all_preds)
    return total_loss / len(loader), metrics


def validate_epoch(model, loader, criterion, device, output_is_logits=True):
    """验证一个epoch"""
    model.eval()
    total_loss = 0
    all_preds, all_labels, all_probs = [], [], []

    pbar = tqdm(loader, desc='Validation', unit='batch', ncols=100)
    with torch.no_grad():
        for img1, img2, labels in pbar:
            img1, img2, labels = img1.to(device), img2.to(device), labels.float().to(device)

            outputs, _, _ = model(img1, img2)

            if output_is_logits:
                loss = criterion(outputs, labels.unsqueeze(1))
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).float()
            else:
                loss = criterion(outputs, labels)
                probs = outputs
                preds = (outputs > 0.5).float()

            total_loss += loss.item()
            all_preds.extend(preds.detach().cpu().numpy())
            all_labels.extend(labels.detach().cpu().numpy())
            all_probs.extend(probs.detach().cpu().numpy())

            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    metrics = compute_metrics(all_labels, all_preds, all_probs)
    return total_loss / len(loader), metrics


def save_model(model, save_path, epoch, optimizer, best_val_f1, model_type, model_config):
    """保存模型"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # 准备要保存的配置信息
    save_dict = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_f1': best_val_f1,
        'model_type': model_type,
        'config': model_config,
    }

    torch.save(save_dict, save_path)
    print(f"模型已保存: {save_path}")


def create_balanced_sampler(dataset, labels_list):
    """创建平衡采样器，确保正负样本被等概率采样"""
    class_sample_count = np.array([len(np.where(labels_list == t)[0]) for t in np.unique(labels_list)])
    weight = 1. / class_sample_count
    samples_weight = np.array([weight[t] for t in labels_list])
    samples_weight = torch.from_numpy(samples_weight).float()
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight), replacement=True)
    return sampler


def create_model(model_type, config):
    """
    根据配置创建模型

    Args:
        model_type: 'lightweight_cnn', 'enhanced_cnn', 'vit', 'lightweight_vit'
        config: 配置对象
    """
    print(f"\n创建模型: {model_type}")

    if model_type == 'lightweight_vit':
        model = LightweightViTSiamese(
            img_size=config.input_size,
            patch_size=getattr(config, 'vit_patch_size', 16),
            embed_dim=getattr(config, 'vit_embed_dim', 256),
            num_heads=getattr(config, 'vit_num_heads', 4),
            num_layers=getattr(config, 'vit_num_layers', 6),
            dropout=getattr(config, 'vit_dropout', 0.1)
        )
        output_is_logits = True  # ViT输出logits
        model_config = {
            'type': 'lightweight_vit',
            'img_size': config.input_size,
            'patch_size': getattr(config, 'vit_patch_size', 16),
            'embed_dim': getattr(config, 'vit_embed_dim', 256),
            'num_heads': getattr(config, 'vit_num_heads', 4),
            'num_layers': getattr(config, 'vit_num_layers', 6),
        }

    elif model_type == 'vit':
        model = ViTSiameseNetwork(
            img_size=config.input_size,
            patch_size=getattr(config, 'vit_patch_size', 16),
            embed_dim=getattr(config, 'vit_embed_dim', 384),
            num_heads=getattr(config, 'vit_num_heads', 6),
            num_layers=getattr(config, 'vit_num_layers', 8),
            dropout=getattr(config, 'vit_dropout', 0.1),
            use_cross_attention=getattr(config, 'vit_use_cross_attention', True),
            aggregation_method=getattr(config, 'vit_aggregation', 'cls')
        )
        output_is_logits = True
        model_config = {
            'type': 'vit',
            'img_size': config.input_size,
            'patch_size': getattr(config, 'vit_patch_size', 16),
            'embed_dim': getattr(config, 'vit_embed_dim', 384),
            'num_heads': getattr(config, 'vit_num_heads', 6),
            'num_layers': getattr(config, 'vit_num_layers', 8),
            'use_cross_attention': getattr(config, 'vit_use_cross_attention', True),
        }

    elif model_type == 'enhanced_cnn':
        model = EnhancedSiameseNetwork(
            backbone=getattr(config, 'backbone', 'resnet18'),
            pretrained=True,
            use_attention=True
        )
        output_is_logits = True
        model_config = {
            'type': 'enhanced_cnn',
            'backbone': getattr(config, 'backbone', 'resnet18'),
        }

    else:  # lightweight_cnn (默认)
        model = LightweightSiameseNetwork(
            backbone=getattr(config, 'backbone', 'mobilenet_v3_small'),
            pretrained=True,
            use_sigmoid=False
        )
        output_is_logits = True
        model_config = {
            'type': 'lightweight_cnn',
            'backbone': getattr(config, 'backbone', 'mobilenet_v3_small'),
        }

    return model, output_is_logits, model_config


def main():
    print("=" * 60)
    print("孪生网络训练脚本（支持CNN/ViT）")
    print("=" * 60)
    print(f"设备: {config.device}")

    # 获取模型类型
    model_type = getattr(config, 'model_type', 'lightweight_vit')
    print(f"模型类型: {model_type}")

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

    # 4. 创建模型
    model, output_is_logits, model_config = create_model(model_type, config)
    model = model.to(config.device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"总参数: {total_params:,}")
    print(f"模型输出格式: {'logits' if output_is_logits else 'probability'}")

    # 5. 计算正负样本权重（解决类别不平衡问题）
    train_labels = [ann['label'] for ann in train_anns]
    label_counts = Counter(train_labels)
    pos_count = label_counts[1]
    neg_count = label_counts[0]

    print(f"\n类别统计:")
    print(f"  正样本数（同一辆车）: {pos_count}")
    print(f"  负样本数（不同车辆）: {neg_count}")
    print(f"  正负比例: {pos_count / neg_count:.2f}")

    # 计算 pos_weight（负样本数/正样本数）
    pos_weight = torch.tensor([neg_count / pos_count]).to(config.device)
    print(f"  pos_weight: {pos_weight.item():.2f}")

    # 6. 使用带权重的 BCEWithLogitsLoss
    criterion = BCEWithLogitsLoss(pos_weight=pos_weight)

    # 7. 优化器（ViT通常需要更小的学习率）
    if 'vit' in model_type:
        learning_rate = getattr(config, 'vit_learning_rate', config.learning_rate * 0.5)
        weight_decay = getattr(config, 'vit_weight_decay', 0.05)
    else:
        learning_rate = config.learning_rate
        weight_decay = 1e-4

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.num_epochs, eta_min=1e-6)

    # 8. 创建平衡采样器
    use_balanced_sampling = getattr(config, 'use_balanced_sampling', True)
    if use_balanced_sampling:
        print("\n使用平衡采样器（正负样本等概率采样）")
        balanced_sampler = create_balanced_sampler(train_dataset, np.array(train_labels))
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size,
                                  sampler=balanced_sampler, num_workers=config.num_workers)
    else:
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size,
                                  shuffle=True, num_workers=config.num_workers)

    val_loader = DataLoader(val_dataset, batch_size=config.batch_size,
                            shuffle=False, num_workers=config.num_workers)

    best_val_f1 = 0
    patience = getattr(config, 'patience', 15)
    no_improve = 0

    # 难负样本挖掘参数
    hard_mining_interval = getattr(config, 'hard_mining_interval', 5)
    hard_threshold = getattr(config, 'hard_negative_threshold', 0.7)
    hard_weight = getattr(config, 'hard_negative_weight', 5.0)
    use_hard_mining = getattr(config, 'use_hard_mining', True)

    print(f"\n开始训练...")
    print(f"  总轮数: {config.num_epochs}")
    print(f"  批次大小: {config.batch_size}")
    print(f"  学习率: {learning_rate}")
    print(f"  难负样本挖掘: {'开启' if use_hard_mining else '关闭'}")
    print("-" * 60)

    for epoch in range(config.num_epochs):
        # ========== 定期挖掘难负样本 ==========
        if use_hard_mining and epoch > 0 and epoch % hard_mining_interval == 0:
            print(f"\n[Epoch {epoch}] 开始难负样本挖掘...")

            # 使用验证集预处理方式创建临时数据集用于挖掘（避免数据增强干扰）
            temp_dataset = TruckPairDataset(train_anns, config.image_root, val_transform)
            hard_neg_indices = mine_hard_negatives(
                model, temp_dataset, config.device,
                hard_threshold=hard_threshold, top_k=50
            )

            if hard_neg_indices:
                print(f"  发现 {len(hard_neg_indices)} 个难负样本，创建加权采样器")
                sampler = create_weighted_sampler(train_dataset, hard_neg_indices, hard_weight)
                train_loader = DataLoader(train_dataset, batch_size=config.batch_size,
                                          sampler=sampler, num_workers=config.num_workers)
            else:
                # 没有难负样本，使用普通平衡采样
                if use_balanced_sampling:
                    train_loader = DataLoader(train_dataset, batch_size=config.batch_size,
                                              sampler=balanced_sampler, num_workers=config.num_workers)
                else:
                    train_loader = DataLoader(train_dataset, batch_size=config.batch_size,
                                              shuffle=True, num_workers=config.num_workers)

        # 训练
        train_loss, train_metrics = train_epoch(model, train_loader, criterion, optimizer,
                                                config.device, output_is_logits)

        # 验证
        val_loss, val_metrics = validate_epoch(model, val_loader, criterion, config.device, output_is_logits)
        scheduler.step()

        print(f"\nEpoch {epoch + 1}/{config.num_epochs} | LR: {optimizer.param_groups[0]['lr']:.6f}")
        print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_metrics['accuracy']:.4f}")
        print(f"  Val   - Loss: {val_loss:.4f}, F1: {val_metrics['f1']:.4f}, AUC: {val_metrics['auc']:.4f}")

        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            no_improve = 0
            save_path = config.save_dir / f'{model_type}_best.pth'
            save_model(model, save_path, epoch, optimizer, best_val_f1, model_type, model_config)
            print(f"  *** 新最佳模型! (F1: {best_val_f1:.4f}) ***")
        else:
            no_improve += 1
            if no_improve >= patience and epoch >= 10:
                print(f"\n早停触发！最佳F1: {best_val_f1:.4f}")
                break

    print("\n训练完成!")
    print(f"最佳验证F1: {best_val_f1:.4f}")


if __name__ == "__main__":
    main()