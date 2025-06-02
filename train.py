# train.py：训练模型并保存最佳权重（动态 Dropout + Label Smoothing + 学习率衰减）

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR
from VGG_BatchNorm.models.vgg import VGG_A_Light # 可切换为 VGG_A 或 VGG_A_Dropout
from VGG_BatchNorm.data.loaders import get_cifar_loader

# ✅ Label Smoothing 损失函数
class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes=10, smoothing=0.1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x, target):
        logprobs = self.log_softmax(x)
        with torch.no_grad():
            true_dist = torch.zeros_like(logprobs)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * logprobs, dim=-1))

# ✅ 设置训练设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ 模型、损失函数、优化器、学习率调度器
model = VGG_A_Light().to(device)  # 替换为 VGG_A() 或 VGG_A_Dropout() 即可
criterion = LabelSmoothingLoss(classes=10, smoothing=0.1)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-3)
scheduler = StepLR(optimizer, step_size=10, gamma=0.5)

# ✅ 加载 CIFAR-10 数据
train_loader = get_cifar_loader(batch_size=128, train=True)
test_loader = get_cifar_loader(batch_size=128, train=False)

# ✅ 测试函数
def evaluate(model):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            pred = outputs.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total

# ✅ 动态 Dropout 控制函数
def set_dropout_probability(model, p):
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.p = p

# ✅ 主训练函数
def train(epochs=20):
    best_acc = 0.0
    train_acc_list = []
    test_acc_list = []
    loss_list = []

    for epoch in range(1, epochs + 1):
        model.train()
        total = 0
        correct = 0
        running_loss = 0.0

        # 每轮动态设置 Dropout 概率（线性从 0.05 增长到 0.5）
        current_dropout = min(0.5, 0.05 + (epoch / epochs) * 0.45)
        set_dropout_probability(model, current_dropout)

        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * X.size(0)
            correct += (outputs.argmax(1) == y).sum().item()
            total += y.size(0)

        scheduler.step()
        train_acc = correct / total
        test_acc = evaluate(model)

        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        loss_list.append(running_loss / total)

        print(f"Epoch {epoch}: Dropout={current_dropout:.2f}, LR={scheduler.get_last_lr()[0]:.6f}, Train Loss={running_loss/total:.4f}, Train Acc={train_acc:.4f}, Test Acc={test_acc:.4f}")

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), "best_model_Light.pth")
            print("✅ 已保存最佳模型")

if __name__ == '__main__':
    train()
