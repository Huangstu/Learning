import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# 生成模拟数据 
torch.manual_seed(42)          # 固定随机种子
true_w = 2.0                   # 真实斜率
true_b = 1.0                   # 真实截距
num_samples = 100

x = torch.rand(num_samples, 1) * 10          # x 在 [0,10) 均匀分布
noise = torch.randn(num_samples, 1) * 0.5    # 高斯噪声
y = true_w * x + true_b + noise              # y = 2x + 1 + 噪声

# 定义模型 
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)        # 输入维度1，输出维度1

    def forward(self, x):
        return self.linear(x)

model = LinearRegressionModel()

# 定义损失函数和优化器 
criterion = nn.MSELoss()                     # 均方误差损失
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)  # 随机梯度下降

# 训练循环 
num_epochs = 500
losses = []

for epoch in range(num_epochs):
    # 前向传播
    y_pred = model(x)
    loss = criterion(y_pred, y)
    losses.append(loss.item())

    # 反向传播
    optimizer.zero_grad()    # 清空梯度
    loss.backward()          # 自动求导
    optimizer.step()         # 更新参数

    # 每 50 轮打印一次
    if (epoch+1) % 50 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# 打印最终参数 
w_learned = model.linear.weight.item()
b_learned = model.linear.bias.item()
print(f'\n真实: w={true_w}, b={true_b}')
print(f'学习: w={w_learned:.4f}, b={b_learned:.4f}')

# 可视化 
plt.figure(figsize=(8,5))
plt.scatter(x.numpy(), y.numpy(), s=10, label='真实数据')
with torch.no_grad():
    x_test = torch.linspace(0, 10, 100).reshape(-1, 1)
    y_test = model(x_test)
plt.plot(x_test.numpy(), y_test.numpy(), 'r-', label='拟合直线')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('线性回归拟合结果')

# 另画损失曲线
plt.figure(figsize=(8,4))
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('训练损失变化')
plt.show()