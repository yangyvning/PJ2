


# VGG_CIFAR10: Training & Visualizing VGG_A / VGG_A_BN on CIFAR-10

本项目基于 PyTorch 实现了 VGG_A、VGG_A_Dropout、VGG_A_BN 等模型在 CIFAR-10 数据集上的训练、可视化、损失函数实验和 loss landscape 分析。支持训练曲线可视化、卷积核可视化、梯度追踪等。



##  项目结构说明

```

VGG\_BatchNorm/
├── models/                     # 模型定义文件
│   ├── vgg.py                  # VGG\_A, VGG\_A\_Dropout, VGG\_A\_BN 等模型
│   └── init.py
├── data/
│   └── loaders.py             # CIFAR-10 加载器
├── utils/
│   └── nn.py                  # 权重初始化工具
├── train.py                   # 主训练脚本（含动态Dropout、LabelSmoothing）
├── train\_debug.py            # 调试版训练脚本（输出中间变量）
├── VGG\_Loss\_Landscape.py     # 可视化 loss landscape 脚本
├── visualize\_kernels.py      # 卷积核可视化脚本
├── requirements.txt
└── README.md

````


##支持的模型与功能
```

| 模型版本            | 描述                 |
| --------------- | ------------------ |
| `VGG_A`         | 原始模型，使用 LeakyReLU  |
| `VGG_A_Light` |   尝试不同神经元 / 卷积核数量 |
| `VGG_A_BN`      | 增加 BatchNorm，收敛更稳定 |
```

###  可选训练增强功能
```
* Label Smoothing Loss
* 动态 Dropout（随 Epoch 增大）
* SGD / Adam 优化器切换
* 权重初始化
* 学习率衰减（StepLR）
```


## 模型训练（以 VGG\_A\_BN 为例）

```
python train.py

训练结束后会自动保存最佳模型至 `best_model_BN.pth`
```

##  Loss Landscape 可视化

用于分析模型训练时 loss 的稳定性与光滑性（对比 BN/非BN）。

```
python VGG_Loss_Landscape.py
```

* 输出图像：`loss_landscape_comparison.png`
* 自动使用多个学习率组合，并提取 Min/Max Loss 曲线

---

## 可视化卷积核

查看第一层卷积核的学习效果：

```
python visualize_kernels.py
```

* 输出图像：`conv1_kernels.png`


## 实验环境
```
| 库           | 版本   |
| ----------- | ------ |
| Python      | 3.9   |
| PyTorch     |2.0.1+cu118 |
| torchvision | 0.15.2+cu118|
| matplotlib  | 3.7.5  |
```
