import torch
print("PyTorch 版本:", torch.__version__)
print("PyTorch 使用的 CUDA 版本:", torch.version.cuda)
print("CUDA 是否可用:", torch.cuda.is_available())

# 尝试在 GPU 上创建一个张量并进行一次简单计算
if torch.cuda.is_available():
    x = torch.randn(100, 100).cuda()
    y = torch.randn(100, 100).cuda()
    z = x @ y
    print("GPU 矩阵乘法成功:", z.device)