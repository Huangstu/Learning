import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

# ---------- 1. 检查 GPU 是否可用 ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# ---------- 2. 数据预处理与数据增强 ----------
# 训练集：随机水平翻转、随机裁剪（填充4像素）、归一化
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet 均值
                         std=[0.229, 0.224, 0.225])   # ImageNet 标准差
])

# 测试集：只进行归一化（不增强）
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# 下载并加载数据
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
test_dataset  = datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)

batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

# 类别名称
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# ---------- 3. 定义 CNN 模型 ----------
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # 卷积层 1: 输入3通道，输出32通道，卷积核3x3，填充1保持尺寸
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        # 卷积层 2: 输入32通道，输出64通道
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # 池化层：2x2 最大池化
        self.pool = nn.MaxPool2d(2, 2)
        # 全连接层：图像经过两次池化后尺寸变为 32/2/2 = 8，所以特征图尺寸为 8x8
        self.fc1 = nn.Linear(64 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
        # Dropout 防止过拟合
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # 卷积 + ReLU + 池化
        x = self.pool(F.relu(self.conv1(x)))   # 32x32 -> 16x16
        x = self.pool(F.relu(self.conv2(x)))   # 16x16 -> 8x8
        # 展平
        x = x.view(-1, 64 * 8 * 8)
        # 全连接层
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)   # 输出层，无 softmax（CrossEntropyLoss 自带）
        return x

model = SimpleCNN().to(device)   # 将模型移至 GPU（如果可用）

# ---------- 4. 损失函数和优化器 ----------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ---------- 5. 训练循环 ----------
num_epochs = 20
train_losses = []
test_accuracies = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)   # 移至 GPU

        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    train_losses.append(avg_loss)

    # ---------- 6. 每个 epoch 结束后评估测试集 ----------
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    test_accuracies.append(accuracy)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Test Acc: {accuracy:.2f}%")

print("训练完成！")

# ---------- 7. 绘制损失和准确率曲线 ----------
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='训练损失')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(test_accuracies, label='测试准确率')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.grid(True)
plt.show()

# ---------- 8. 可视化部分预测结果 ----------
# 从测试集中取一批图像（已归一化，需反归一化显示）
data_iter = iter(test_loader)
images, labels = next(data_iter)
images, labels = images.to(device), labels.to(device)

model.eval()
with torch.no_grad():
    outputs = model(images)
    _, preds = torch.max(outputs, 1)

# 反归一化（用于显示）
inv_normalize = transforms.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
    std=[1/0.229, 1/0.224, 1/0.225]
)

fig, axes = plt.subplots(2, 4, figsize=(10, 5))
axes = axes.flatten()
for i in range(8):
    img = images[i].cpu().clone()
    img = inv_normalize(img)   # 反归一化
    img = img.permute(1, 2, 0).numpy()   # 转为 (H, W, C)
    # 裁剪到 [0,1] 范围（防止数值溢出）
    img = np.clip(img, 0, 1)
    ax = axes[i]
    ax.imshow(img)
    ax.set_title(f'真实: {classes[labels[i]]}\n预测: {classes[preds[i]]}')
    ax.axis('off')
plt.tight_layout()
plt.show()

# 保存模型（可选）
# torch.save(model.state_dict(), 'cifar10_cnn.pth')