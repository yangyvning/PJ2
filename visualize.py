

# visualize.py：卷积核可视化 + OpenMP 冲突兼容

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # 避免 OpenMP 多重加载错误

import torch
import matplotlib.pyplot as plt
from VGG_BatchNorm.models.vgg import VGG_A_BN  # 如果你是用 VGG_A 训练的就改这里

# 加载模型权重
model = VGG_A_BN()
model.load_state_dict(torch.load("best_model_BN.pth", map_location="cpu"))
model.eval()

# 获取第一层卷积核
filters = model.features[0].weight.data.clone()

# 可视化前 16 个卷积核
plt.figure(figsize=(8, 8))
for i in range(16):
    f = filters[i]
    f = (f - f.min()) / (f.max() - f.min())  # 归一化到 0~1
    plt.subplot(4, 4, i+1)
    plt.imshow(f.permute(1, 2, 0))  # CHW -> HWC
    plt.axis('off')

plt.tight_layout()
plt.savefig("filters_BN.png")
print("✅ 卷积核图已保存为 filters_BN.png")
