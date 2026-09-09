import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


plt.rcParams['font.sans-serif'] = ['SimHei']  
plt.rcParams['axes.unicode_minus'] = False    

# 生成非线性数据（二次函数 + 噪声） 
torch.manual_seed(42)
num_samples = 150
x = torch.rand(num_samples, 1) * 6 - 3          # x 在 [-3, 3)
# 真实多项式系数: y = 2.0*x^2 - 1.2*x + 0.5
true_coeffs = [0.5, -1.2, 2.0]                  # 对应 [常数项, x, x^2]

# 手动构造多项式特征矩阵 [1, x, x^2]
X_poly = torch.cat([torch.ones(num_samples, 1), x, x**2], dim=1)
# 计算真实 y，加噪声
y = X_poly @ torch.tensor(true_coeffs).reshape(-1, 1) + torch.randn(num_samples, 1) * 0.3

# 定义模型（线性层，输入维度为3） 
class PolyRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 1)   # 输入3个特征（偏置项、x、x^2）

    def forward(self, x):
        return self.linear(x)

model = PolyRegressionModel()

#  损失函数和优化器 
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.05)  # Adam 收敛更快

# 训练循环 
epochs = 800
for epoch in range(epochs):
    y_pred = model(X_poly)          # 直接输入特征矩阵
    loss = criterion(y_pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}')


# 注意模型自带一个偏置（bias），而我们数据中已经包含常数项列（全1），
# 所以最终预测 = w1*1 + w2*x + w3*x^2 + bias。
# 为了方便对比真实系数，将 bias 合并到常数项系数中。
w_learned = model.linear.weight.detach().numpy().flatten()   # 形状 (3,)
b_learned = model.linear.bias.item()
print(f"\n学到的权重（不含bias）: {w_learned}")
print(f"学到的bias: {b_learned:.4f}")

# 合并 bias 到常数项
final_coeffs = w_learned.copy()
final_coeffs[0] += b_learned
print(f"合并后的系数 [常数, x, x^2]: {final_coeffs}")
print(f"真实系数: {true_coeffs}")

# 保存模型 
torch.save(model.state_dict(), 'poly_model.pth')
print("\n模型已保存为 poly_model.pth")

# 加载模型并预测新数据 
new_model = PolyRegressionModel()
new_model.load_state_dict(torch.load('poly_model.pth'))
new_model.eval()   # 切换到评估模式

# 预测 x = 1.5 时的 y
x_new = torch.tensor([[1.5]])
# 构造多项式特征
X_new = torch.cat([torch.ones(1, 1), x_new, x_new**2], dim=1)
with torch.no_grad():
    pred = new_model(X_new)
print(f"\n预测 x=1.5 时，y = {pred.item():.4f}")

# 可视化拟合曲线 
plt.figure(figsize=(8, 5))
plt.scatter(x.numpy(), y.numpy(), s=10, label='真实数据')
with torch.no_grad():
    x_plot = torch.linspace(-3, 3, 200).reshape(-1, 1)
    X_plot = torch.cat([torch.ones(200, 1), x_plot, x_plot**2], dim=1)
    y_plot = new_model(X_plot)
plt.plot(x_plot.numpy(), y_plot.numpy(), 'r-', label='拟合曲线')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('多项式回归拟合结果')
plt.show()