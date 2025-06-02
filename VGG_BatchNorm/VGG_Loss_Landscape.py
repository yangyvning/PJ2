import sys
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.append(os.path.abspath("."))

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
import random
from tqdm import tqdm

from models.vgg import VGG_A, VGG_A_BN
from data.loaders import get_cifar_loader

# -------------------------------
# 设置设备和随机种子
# -------------------------------
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

def set_random_seeds(seed_value=0, device='cpu'):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    random.seed(seed_value)
    if device != 'cpu':
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# -------------------------------
# 准确率评估函数
# -------------------------------
def get_accuracy(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            preds = outputs.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total

# -------------------------------
# 单模型训练函数
# -------------------------------
def train(model, optimizer, criterion, train_loader, val_loader, epochs_n=20):
    model.to(device)
    losses_list = []

    for epoch in tqdm(range(epochs_n), desc="Epoch"):
        model.train()
        epoch_loss = []

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            epoch_loss.append(loss.item())
            loss.backward()
            optimizer.step()

        losses_list.append(epoch_loss)

    return losses_list

# -------------------------------
# 提取所有 lr 下每轮 epoch 的 max/min loss 曲线
# -------------------------------
def extract_true_min_max_curves(losses_all_lr):
    min_curve = []
    max_curve = []
    num_epochs = len(losses_all_lr[0])  # 假设每个模型训练轮数一致
    for epoch in range(num_epochs):
        epoch_losses = [np.mean(losses_lr[epoch]) for losses_lr in losses_all_lr]
        min_curve.append(min(epoch_losses))
        max_curve.append(max(epoch_losses))
    return min_curve, max_curve

# -------------------------------
# 多学习率下训练不同模型，收集 loss
# -------------------------------
def multi_lr_loss_collection(model_class):
    lrs  =  [0.01, 0.02, 0.005, 0.001]

    criterion = nn.CrossEntropyLoss()
    losses_all = []

    for lr in lrs:
        model = model_class()
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        losses_list = train(model, optimizer, criterion, train_loader, val_loader, epochs_n=20)
        losses_all.append(losses_list)
    return losses_all

# -------------------------------
# 绘制最终曲线图
# -------------------------------
def plot_loss_landscape_comparison(min1, max1, min2, max2):
    plt.figure(figsize=(10, 5))
    epochs = list(range(1, len(min1) + 1))

    plt.plot(epochs, min1, label='VGG_A Min Loss', color='blue', linestyle='--')
    plt.plot(epochs, max1, label='VGG_A Max Loss', color='blue')
    plt.fill_between(epochs, min1, max1, color='blue', alpha=0.2)

    plt.plot(epochs, min2, label='VGG_A_BN Min Loss', color='green', linestyle='--')
    plt.plot(epochs, max2, label='VGG_A_BN Max Loss', color='green')
    plt.fill_between(epochs, min2, max2, color='green', alpha=0.2)

    plt.title("Loss Landscape Comparison: VGG_A vs VGG_A_BN")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig("loss_landscape_comparison.png")
    print("✅ 已保存图像：loss_landscape_comparison.png")

# -------------------------------
# 主程序
# -------------------------------
if __name__ == '__main__':
    set_random_seeds(2020, device)

    # 加载数据
    train_loader = get_cifar_loader(batch_size=128, train=True, num_workers=0)
    val_loader = get_cifar_loader(batch_size=128, train=False, num_workers=0)

    # VGG_A（无BN）
    print("▶️ 正在训练 VGG_A（无 BN）...")
    losses_vgg = multi_lr_loss_collection(VGG_A)

    # VGG_A_BN（有BN）
    print("▶️ 正在训练 VGG_A_BatchNorm（有 BN）...")
    losses_bn = multi_lr_loss_collection(VGG_A_BN)

    # 提取 min/max 曲线
    min_vgg, max_vgg = extract_true_min_max_curves(losses_vgg)
    min_bn,  max_bn  = extract_true_min_max_curves(losses_bn)

    # 绘制 loss landscape 比较图
    plot_loss_landscape_comparison(min_vgg, max_vgg, min_bn, max_bn)
