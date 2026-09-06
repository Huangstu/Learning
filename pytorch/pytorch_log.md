
#### 第一章：PyTorch 简介与安装


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

---

#### 第二章：PyTorch 基础知识

##### 2.1 张量（Tensor）
多维数组，支持 GPU 加速。

###### 2.1.1 创建张量
**从列表创建**
```python
import torch
x = torch.tensor([[1, 2], [3, 4]])
print(x)
```
**特殊值张量**
```python
zeros = torch.zeros(3, 4)
ones = torch.ones(2, 2, dtype=torch.int16)
```
**随机张量**
```python
rand_tensor = torch.rand(2, 2)      # [0,1) 均匀分布
torch.manual_seed(42)               # 固定随机种子
```

###### 2.1.2 张量属性
```python
tensor = torch.rand(3, 4)
print(tensor.shape, tensor.dtype, tensor.device)
```

###### 2.1.3 基本运算
**索引与切片**
```python
t = torch.tensor([[1,2,3],[4,5,6]])
print(t[0,1])    # 2
print(t[0,:])    # [1,2,3]
```
**算术运算**
```python
a = torch.tensor([1,2,3])
b = torch.tensor([4,5,6])
print(a + b)           # 加法
print(a * b)           # 逐元素乘
c = torch.tensor([[1,2],[3,4]])
d = torch.tensor([[5,6],[7,8]])
print(torch.mm(c,d))   # 矩阵乘
```
**形状变换**
```python
t = torch.rand(4,4)
print(t.reshape(2,8).shape)  # torch.Size([2,8])
```

###### 2.1.4 与 NumPy 互转
```python
import numpy as np
np_arr = np.array([1,2,3])
torch_tensor = torch.from_numpy(np_arr)   # NumPy → Torch
torch_tensor = torch.tensor([4,5,6])
np_arr = torch_tensor.numpy()             # Torch → NumPy
```

##### 2.2 自动求导（Autograd）
自动计算梯度，核心机制是**计算图**。

###### 2.2.1 requires_grad 属性
设置 `requires_grad=True` 追踪操作。

###### 2.2.2 计算梯度
调用 `.backward()`，梯度存入 `.grad`。
```python
x = torch.tensor(2.0, requires_grad=True)
y = x ** 2
y.backward()
print(x.grad)  # 4.0
```
```python
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
z = x * 2
loss = z.sum()
loss.backward()
print(x.grad)  # [2., 2., 2.]
```

###### 2.2.3 关闭梯度追踪（推理阶段）
```python
with torch.no_grad():
    y = x * 3   # 不记录 grad_fn
```
