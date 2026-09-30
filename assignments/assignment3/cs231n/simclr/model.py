import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet50
# 一个50层的残差网络

class Model(nn.Module):
    def __init__(self, feature_dim=128):
        super(Model, self).__init__()

        # 初始化一个空列表，用于暂存编码器（encoder）的网络层
        self.f = []

        # 遍历 resnet50() 的直接子模块（named_children），name 是模块名，module 是对应的网络层
        for name, module in resnet50().named_children():
            # 找到名为 'conv1' 的原始输入卷积层
            if name == 'conv1':
                # 替换原有的 7x7 步长2卷积，改为 3x3 步长1、padding 1 且无偏置的卷积
                # 这是因为 CIFAR-10 图像只有 32x32，原版下采样太狠，会导致空间信息急剧丢失
                module = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            
            # 过滤掉原版 ImageNet 分类用的全连接层（nn.Linear）和最大池化层（nn.MaxPool2d）
            if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d):
                # 将保留的层追加到 self.f 列表中
                self.f.append(module)

        # 将列表解包并组成一个 nn.Sequential 容器，作为基础编码器 f
        self.f = nn.Sequential(*self.f)
        
        # 构建投影头 g，使用 nn.Sequential 依次封装四层
        # 第一层 nn.Linear 把 2048 维的 h 压缩到 512 维，因为紧跟 BatchNorm 去掉了 bias
        # nn.BatchNorm1d 对 512 维特征做批归一化，稳定训练并加速收敛
        # nn.ReLU 引入非线性激活，inplace=True 表示原地操作以节省显存
        # 第二层 nn.Linear 把 512 维映射到 feature_dim（默认128）的目标对比空间，保留 bias
        self.g = nn.Sequential(nn.Linear(2048, 512, bias=False), nn.BatchNorm1d(512),
                               nn.ReLU(inplace=True), nn.Linear(512, feature_dim, bias=True))

    def forward(self, x):
        # 将输入图像 x 送入基础编码器 f，得到高维空间特征图
        x = self.f(x)
        # 从第 1 维（通道维度）开始展平特征图，得到表示向量 h，形状为 [Batch, 2048*H*W]
        feature = torch.flatten(x, start_dim=1)
        # 将 h 送入投影头 g，得到映射到对比空间的向量 z，形状为 [Batch, feature_dim]
        out = self.g(feature)
        
        # 分别对 h 和 z 在最后一维做 L2 归一化，使它们的模长为 1
        # 归一化是为了让后续计算余弦相似度时，点积结果直接等于余弦值
        return F.normalize(feature, dim=-1), F.normalize(out, dim=-1)