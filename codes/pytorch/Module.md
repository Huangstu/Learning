
#### Module父类常用函数


```python
import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self):
        # 必须最先调用！初始化父类的内部状态（如参数注册表、模块注册表）
        super().__init__() 
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 32 * 32, 10) # 假设维度

    def forward(self, x):
        # 最核心的方法！你必须要重写它来定义数据流向。
        # 注意：不要直接调 model.forward(x)，而是直接写 model(x)。
        # 因为父类的 __call__ 方法会自动调用 forward。
        return self.fc1(x)

# ============================
# 实例化模型
# ============================
model = SimpleCNN()

# 【1. 设备迁移】
model.to('cuda') # 将模型的所有参数和缓冲区（Buffer）移动到GPU
# 等价于: model.cuda() 
# 也可以用: model.cpu() 移回CPU

# 【2. 模式切换】(用于控制Dropout和BatchNorm)
model.train() # 训练模式（启用Dropout，更新BN的统计量）
model.eval()  # 评估/测试模式（关闭Dropout，使用固定的BN统计量）

# 【3. 参数管理】(交给优化器)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# model.named_parameters() 可以返回名字和参数，方便打印调试
for name, param in model.named_parameters():
    print(name, param.shape)

# 【4. 梯度清零】(防止梯度累加)
model.zero_grad() # 等价于遍历所有参数执行 param.grad = None
# 通常在训练循环里会用 optimizer.zero_grad()，两者效果一样

# 【5. 模型保存与加载】(最重要的功能！)
# 保存权重字典（推荐保存这个，不包含网络结构）
torch.save(model.state_dict(), 'model.pth')

# 加载权重
model.load_state_dict(torch.load('model.pth'))

# 【6. 遍历模块】(用于初始化权重或微调)
# model.modules() 递归遍历所有的子模块（包括最外层）
for module in model.modules():
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight) # 例如：手动初始化卷积层权重

# model.children() 只遍历直接子层（不递归）

# 【7. 应用函数】(非常优雅的初始化方法)
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        
model.apply(init_weights) # 父类会递归地把这个函数应用到每一个子模块上

# 【8. 设置梯度计算开关】
model.requires_grad_(False) # 整个模型冻结（比如迁移学习时）
model.requires_grad_(True)  # 解冻

# 【9. 查看模型的字符串表示】(自动调用 __repr__)
print(model) 
```