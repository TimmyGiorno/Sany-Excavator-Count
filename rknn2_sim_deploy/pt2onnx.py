import types
import torch
from ultralytics import YOLO

print("正在加载模型 best.pt ...")
model = YOLO("best.pt")

def headless_forward(self, x):
    """
    强行重写 Detect 层的推理逻辑：不计算坐标映射，不计算 Sigmoid，不拼接。
    只把边界框特征 (cv2) 和分类特征 (cv3) 拼接在一起后原封不动地返回。
    """
    y = []
    for i in range(self.nl): # 遍历三个尺度的特征图
        # 拼接并在通道维度 (dim=1) 组合
        y.append(torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1))
    return tuple(y)

# 找到模型的最后一层 (Detect 头)
detect_head = model.model.model[-1]
# 强制将它的 forward 函数替换成我们的砍头版
detect_head.forward = types.MethodType(headless_forward, detect_head)
# 骗过原版库的检测机制，防止它再次强加后处理
detect_head.export = False
# ==================================================

print("正在导出纯净版 ONNX ...")
model.export(format="onnx", imgsz=640, opset=12, simplify=True)

print("导出完成！")
