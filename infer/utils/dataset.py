# utils/dataset.py
import json
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset


class TruckPairDataset(Dataset):
    """卡车图片对数据集"""

    def __init__(self, annotations, image_root, transform=None):
        self.annotations = annotations
        self.image_root = Path(image_root)
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        item = self.annotations[idx]
        img1_path = self.image_root / Path(item['img1_path']).name
        img2_path = self.image_root / Path(item['img2_path']).name
        img1 = Image.open(img1_path).convert('RGB')
        img2 = Image.open(img2_path).convert('RGB')
        label = item['label']

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, label


def load_annotations(annotation_file):
    """加载标注文件"""
    with open(annotation_file, 'r', encoding='utf-8') as f:
        annotations = json.load(f)
    return annotations