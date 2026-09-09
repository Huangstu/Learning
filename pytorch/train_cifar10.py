import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter

#import os
# 强制设置 torchvision 下载路径
#os.environ['TORCH_HOME'] = os.path.join(os.getcwd(), 'data')

# 超参数 
batch_size = 64
learning_rate = 0.001
num_epochs = 20
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   #当前是cuda
print(f"使用设备: {device}")

# 数据加载 
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),       #随机剪裁出32x32
    transforms.RandomHorizontalFlip(),          #随机水平翻转
    transforms.ToTensor(),      #图片像素（0-255）变成 PyTorch 能算的浮点数（0.0-1.0）
                                #并调整维度为 (通道, 高, 宽)
    transforms.Normalize(mean=[0.485, 0.456, 0.406],    
                         std=[0.229, 0.224, 0.225])
                                #标准化。
                                #用这组均值和标准差将数据变成“均值为0，方差为1”的标准正态分布
                                #是 ImageNet 数据集的统计结果
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
#不能做随机裁剪和翻转（测试必须公平）
])

train_dataset = datasets.CIFAR10('./data', train=True, download=False, transform=transform_train)
test_dataset  = datasets.CIFAR10('./data', train=False, download=False, transform=transform_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)    
#[Batch, Channel, Height, Width]                                                                                                
#num_workers表示加载数据时使用的子进程数，
#0表示不使用子进程，直接在主进程中加载数据。

#  模型定义 
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        #卷积层
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)     #输入通道数3，输出通道数32，卷积核大小3x3，padding=1保持尺寸不变（图像四周补0）
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)    #继续卷积
        #池化层
        self.pool = nn.MaxPool2d(2, 2)      #最大池化层，2×2的窗口，步长2。把长和宽缩小一半

        #全连接层
        self.fc1 = nn.Linear(64 * 8 * 8, 256)
        #经过两次卷积：3通道 -> 32通道 -> 64通道，所以通道数是 64
        #经过两次池化：32 -> 16 -> 8，所以长宽是 8
        #特征压缩，保留256
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

        self.dropout = nn.Dropout(0.5)  #防作弊机制
        #在训练时，它每次都会随机扔掉 50% 的神经元（把它们变成0），强迫网络不依赖某个特定的神经元， 
        #从而提高模型的泛化能力。测试时则不会扔掉神经元。

    def forward(self, x):
        #激活函数使用ReLU（Rectified Linear Unit，线性整流函数），它的公式是 f(x) = max(0, x)，
        #即小于0的值都变成0，大于0的值保持不变
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        #Flattening 
        x = x.view(-1, 64 * 8 * 8)
        #全连接层只认识一维的向量（数字排列）
        #-1表示自动计算这个维度的大小，保持总元素数量不变
        #经过两次卷积和池化后，特征图的尺寸是 64×8×8，所以要展平成一维向量，长度是 64*8*8=4096

        x = nn.functional.relu(self.fc1(x))
        x = self.dropout(x)
        x = nn.functional.relu(self.fc2(x))
        x = self.dropout(x)
        #输出层
        x = self.fc3(x)
        return x

model = SimpleCNN().to(device)

# 损失函数、优化器、学习率调度
criterion = nn.CrossEntropyLoss() #损失函数，它负责“打分”。接收模型输出的10个分数（Logits）和真实的标签，算出一个数字（Loss）
optimizer = optim.Adam(model.parameters(), lr=learning_rate)    #优化器，用于更新模型参数（先前定义的）
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1) #每跑完10个Epoch（step_size=10），就把学习率乘以 gamma=0.1）

# TensorBoard 
writer = SummaryWriter('runs/cifar10_cnn')

# 训练函数 
def train_one_epoch(epoch):
    #每个 epoch 都要清空 running_loss、correct、total 这些变量，重新统计当前 epoch 的损失和准确率。
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    #每进入一个 batch，都会调用 model(inputs) 进行前向传播，计算损失 loss，然后反向传播更新参数。
    for i, (inputs, labels) in enumerate(train_loader):
        #inputs 的形状是 [batch_size, 3, 32, 32]，表示一个 batch 中有 batch_size 张图片，每张图片有 3 个通道（RGB），大小是 32x32。
        #labels 的形状是 [batch_size]，表示每张图片对应的标签（真实类别）。
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs) #调用model(inputs) 时，会自动调用 SimpleCNN 类的 forward 方法，进行前向传播，得到输出 logits。
        loss = criterion(outputs, labels) #数字越小，表示模型的预测越准确

        optimizer.zero_grad()   #清空梯度缓存。PyTorch 中的梯度是累加的，所以每次反向传播前都要清空梯度，否则会把上一次的梯度也加进来。
        loss.backward()         #反向传播，计算梯度。PyTorch 会根据计算图自动计算每个参数的梯度。
        optimizer.step()        #更新参数。优化器会根据计算得到的梯度和学习率，更新模型的参数。
                                #公式：w = w - lr * dw，其中 w 是参数，lr 是学习率，dw 是梯度。
        running_loss += loss.item()             #.item() 方法可以把一个只有一个元素的张量转换成 Python 的数字。
        _, predicted = torch.max(outputs, 1)    #模型输出的10个分数里，找到最大值的索引，作为预测的类别
        total += labels.size(0)
        correct += (predicted == labels).sum().item()   #正确预测的数量。predicted == labels 会返回一个布尔张量，表示每个预测是否正确。
                                                        #sum() 会把 True 当作 1，False 当作 0，加起来就是正确预测的数量。

        if (i + 1) % 200 == 0:
            print(f"Epoch {epoch+1}, Step {i+1}/{len(train_loader)}, Loss: {loss.item():.4f}")

    avg_loss = running_loss / len(train_loader)
    acc = 100 * correct / total
    print(f"Epoch {epoch+1} 训练平均损失: {avg_loss:.4f}, 准确率: {acc:.2f}%")
    writer.add_scalar('Loss/train', avg_loss, epoch)
    writer.add_scalar('Accuracy/train', acc, epoch)
    return avg_loss

#  验证函数 
def validate(epoch):
    model.eval()    #切换到评估模式，关闭 dropout 和 batch normalization 的训练行为
    correct = 0
    total = 0
    with torch.no_grad():   #关闭梯度计算
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = 100 * correct / total
    print(f"Epoch {epoch+1} 验证准确率: {acc:.2f}%")
    writer.add_scalar('Accuracy/test', acc, epoch)
    return acc


if __name__ == '__main__':
    # 主训练循环
    best_acc = 0.0
    for epoch in range(num_epochs):
        print(f"\n-------- Epoch {epoch+1}/{num_epochs} --------")
        train_loss = train_one_epoch(epoch) 
        test_acc = validate(epoch)  

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"✓ 保存最佳模型 (准确率: {best_acc:.2f}%)")

        scheduler.step()    #更新学习率。每个 epoch 结束后，调用 scheduler.step()，根据预设的策略调整学习率。

    print(f"\n训练完成！最佳测试准确率: {best_acc:.2f}%")
    writer.close()