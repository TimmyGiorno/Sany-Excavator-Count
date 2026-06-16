import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
infer_dir = os.path.join(os.path.dirname(current_dir), 'infer')

if infer_dir not in sys.path:
    sys.path.insert(0, infer_dir)

import torch
import torch.nn as nn

from siamese_model.siamese_network import load_siamese_model


class SiameseFeatureExtractor(nn.Module):
    def __init__(self, siamese_model):
        super(SiameseFeatureExtractor, self).__init__()
        self.model = siamese_model

    def forward(self, x):
        # 只调用 forward_single 提取嵌入向量
        return self.model.forward_single(x)


def export_to_onnx():
    model_path = "./tmp_files/attention_siamese_best.pth"
    onnx_path = "./tmp_files/siamese_extractor.onnx"

    print("--> 正在加载 PyTorch 模型...")
    original_model, _ = load_siamese_model(model_path, device='cpu')

    # 3. 包装模型并设置为推理模式
    extractor = SiameseFeatureExtractor(original_model)
    extractor.eval()

    # 4. 创建虚拟输入 (BatchSize=1, Channel=3, 224x224)
    dummy_input = torch.randn(1, 3, 224, 224)

    # 5. 导出 ONNX
    print("--> 正在导出单分支 ONNX 模型...")
    torch.onnx.export(
        extractor,
        dummy_input,
        onnx_path,
        opset_version=12,
        input_names=["input_image"],
        output_names=["embedding"],
        dynamic_axes={"input_image": {0: "batch_size"}, "embedding": {0: "batch_size"}}
    )
    print(f"--> ✅ 导出成功！保存至: {onnx_path}")


if __name__ == '__main__':
    export_to_onnx()
