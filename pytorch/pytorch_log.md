#### 第一章：PyTorch 基础

##### 1.1 PyTorch 简介
- 基于 Python 的科学计算库，核心功能：
  1. **张量（Tensor）**：类似 NumPy，支持 GPU 加速。
  2. **自动求导（Autograd）**：神经网络训练的核心。
- 特点：**动态计算图**，构建和调试更灵活。

##### 1.2 PyTorch 安装
**CPU 版本**
```bash
pip install torch torchvision torchaudio
```
**CUDA 11.8 版本**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
**验证安装**
```python
import torch
print(torch.__version__)          # 版本号
print(torch.cuda.is_available())  # CUDA 是否可用
```

##### 1.3 张量（Tensor）
多维数组，支持 GPU 加速。

###### 1.3.1 创建张量
```python
import torch
# 从列表
x = torch.tensor([[1, 2], [3, 4]])
# 特殊值
zeros = torch.zeros(3, 4)
ones = torch.ones(2, 2, dtype=torch.int16)
# 随机
rand_tensor = torch.rand(2, 2)      # [0,1) 均匀分布
torch.manual_seed(42)               # 固定随机种子
```

###### 1.3.2 张量属性
```python
tensor = torch.rand(3, 4)
print(tensor.shape, tensor.dtype, tensor.device)
```

###### 1.3.3 基本运算
```python
t = torch.tensor([[1,2,3],[4,5,6]])
print(t[0,1], t[0,:])          # 索引与切片
a, b = torch.tensor([1,2,3]), torch.tensor([4,5,6])
print(a+b, a*b)                # 逐元素加、乘
c, d = torch.tensor([[1,2],[3,4]]), torch.tensor([[5,6],[7,8]])
print(torch.mm(c,d))           # 矩阵乘法
t = torch.rand(4,4)
print(t.reshape(2,8).shape)    # 形状变换
```

###### 1.3.4 与 NumPy 互转
```python
import numpy as np
np_arr = np.array([1,2,3])
torch_tensor = torch.from_numpy(np_arr)   # NumPy → Torch
torch_tensor = torch.tensor([4,5,6])
np_arr = torch_tensor.numpy()             # Torch → NumPy
```

##### 1.4 自动求导（Autograd）
核心是计算图，记录操作以反向传播梯度。

###### 1.4.1 `requires_grad` 属性
```python
x = torch.tensor(2.0, requires_grad=True)
```

###### 1.4.2 计算梯度
```python
y = x ** 2
y.backward()
print(x.grad)  # 4.0
```
向量示例：
```python
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
z = x * 2
loss = z.sum()
loss.backward()
print(x.grad)  # [2., 2., 2.]
```

###### 1.4.3 关闭梯度追踪（推理阶段）
```python
with torch.no_grad():
    y = x * 3   # 不记录 grad_fn
```

---

#### 第二章：数据加载与训练流程模板

##### 2.1 数据集（Dataset）与数据加载器（DataLoader）
PyTorch 使用 `Dataset` 封装数据，`DataLoader` 提供批量迭代。

###### 2.1.1 自定义 Dataset（示例）
```python
from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# 使用 DataLoader
dataset = MyDataset(x, y)
loader = DataLoader(dataset, batch_size=32, shuffle=True)
```

###### 2.1.2 torchvision 内置数据集
```python
from torchvision import datasets, transforms
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
```

##### 2.2 通用训练模板
```python
model = ...               # 定义模型
criterion = ...           # 损失函数
optimizer = ...           # 优化器
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    avg_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")
```

##### 2.3 评估与推理
```python
model.eval()
correct = total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print(f"Accuracy: {100*correct/total:.2f}%")
```

---

#### 第三章：典型项目

##### 3.1 线性回归

###### 目标
一元线性回归 `y = w*x + b`。

###### 核心代码
```python
import torch.nn as nn
import matplotlib.pyplot as plt

torch.manual_seed(42)
true_w, true_b = 2.0, 1.0
x = torch.rand(100, 1) * 10
y = true_w * x + true_b + torch.randn(100, 1) * 0.5

model = nn.Linear(1, 1)
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for _ in range(500):
    y_pred = model(x)
    loss = criterion(y_pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print(f"w={model.weight.item():.4f}, b={model.bias.item():.4f}")
```

###### 关键点
- 使用 `nn.Linear` 作为模型。
- `MSELoss` 和 `SGD` 优化器。
- 训练四步：前向、损失、反向、更新。

##### 3.2 多项式回归与模型保存/加载

###### 目标
拟合二次函数，学会保存/加载模型。

###### 核心代码
```python
# 构造多项式特征 [1, x, x²]
X_poly = torch.cat([torch.ones(100,1), x, x**2], dim=1)
model = nn.Linear(3, 1)
optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
# 训练循环略（同上）
# 保存
torch.save(model.state_dict(), 'poly_model.pth')
# 加载
new_model = nn.Linear(3, 1)
new_model.load_state_dict(torch.load('poly_model.pth'))
new_model.eval()
# 预测新样本
x_new = torch.tensor([[1.5]])
X_new = torch.cat([torch.ones(1,1), x_new, x_new**2], dim=1)
pred = new_model(X_new)
```

###### 关键点
- 手动构造多项式特征，用线性拟合非线性。
- 使用 `state_dict` 保存/加载参数。
- `eval()` 用于推理模式。

##### 3.3 MNIST 全连接分类

###### 目标
三层全连接网络分类手写数字。

###### 核心代码
```python
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(-1))
])
dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
loader = DataLoader(dataset, batch_size=64, shuffle=True)

class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = SimpleNN()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# 训练循环略
# 测试准确率
correct = (predicted == labels).sum().item() / len(labels)
```

###### 关键点
- `ToTensor` 归一化，`Lambda` 展平。
- `CrossEntropyLoss` 自带 softmax。
- 全连接层多分类。

##### 3.4 CIFAR-10 CNN 分类（GPU + 数据增强）

###### 目标
简单 CNN 实现 10 分类，使用 GPU 和数据增强。

###### 核心代码
```python
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64*8*8, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.5)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))   # 32→16
        x = self.pool(F.relu(self.conv2(x)))   # 16→8
        x = x.view(-1, 64*8*8)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

model = SimpleCNN().to(device)
# 训练中 images, labels = images.to(device), labels.to(device)
# 保存最佳模型
best_acc = 0.0
# 每个epoch后 if acc > best_acc: best_acc=acc; torch.save(model.state_dict(), 'best.pth')
```

###### 关键点
- `device` 统一管理硬件。
- 数据增强：翻转和随机裁剪。
- 归一化使用 ImageNet 统计量。
- 卷积+池化+全连接，Dropout 防过拟合。
- 简单 CNN 可达 75%~80% 准确率。

