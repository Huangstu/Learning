import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

# 设置随机种子 
torch.manual_seed(42)

# 数据预处理和加载 
# 将图像从 (28,28) 展平为 784 维向量，并归一化到 [0,1]
transform = transforms.Compose([
    transforms.ToTensor(),                     # 转为 [0,1] 的 Tensor
    transforms.Lambda(lambda x: x.view(-1))   # 展平为 784 维
])

# 下载训练集和测试集
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset  = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# 创建 DataLoader
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 定义网络结构 
class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)   # 输入层 → 隐藏层
        self.fc2 = nn.Linear(128, 64)    # 隐藏层 → 隐藏层
        self.fc3 = nn.Linear(64, 10)     # 隐藏层 → 输出层（10个类别）

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)                  # 注意：没有 softmax，因为 CrossEntropyLoss 自带
        return x

model = SimpleNN()

# 损失函数和优化器
criterion = nn.CrossEntropyLoss()          # 交叉熵损失（包含 softmax）
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练循环 
num_epochs = 10
train_losses = []
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    train_losses.append(avg_loss)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

# 测试准确率 
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)   # 取最大值的索引
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"\n测试准确率: {accuracy:.2f}%")

#  可视化部分预测结果 
# 随机取 6 张测试图像
data_iter = iter(test_loader)
images, labels = next(data_iter)
model.eval()
with torch.no_grad():
    outputs = model(images)
    _, preds = torch.max(outputs, 1)

# 显示图像（这里需将展平数据还原为 28x28）
fig, axes = plt.subplots(2, 3, figsize=(8, 6))
axes = axes.flatten()
for i in range(6):
    img = images[i].numpy().reshape(28, 28)
    ax = axes[i]
    ax.imshow(img, cmap='gray')
    ax.set_title(f'真实: {labels[i]}, 预测: {preds[i]}')
    ax.axis('off')
plt.tight_layout()
plt.show()

# 绘制训练损失曲线
plt.figure(figsize=(6, 4))
plt.plot(train_losses, marker='o')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('训练损失变化')
plt.grid(True)
plt.show()