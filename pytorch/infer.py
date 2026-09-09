import torch
import os
from PIL import Image
from torchvision import transforms
from train_cifar10 import SimpleCNN  # 从之前的代码中导入模型类

# 1. 定义 CIFAR-10 的10个类别标签,按顺序给出
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
           'dog', 'frog', 'horse', 'ship', 'truck']

# 2. 设置设备，与训练保持一致
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 3. 加载模型结构并加载保存好的权重
model = SimpleCNN().to(device)
#model.load_state_dict(torch.load('best_model.pth', map_location=device))
model.load_state_dict(torch.load('best_model.pth', map_location=device, weights_only=True))
model.eval()  # 评估模式

# 4. 定义预处理（必须与训练时测试集的 transform 完全一致，不能有增强！）
# 对于单张图片，我们需要手动调整尺寸
transform = transforms.Compose([
    transforms.Resize((32, 32)),  # 保持一致
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# 5. 定义预测函数
def predict_image(image_path):
    try:
        # 打开图片，并强制转换为RGB（防止遇到RGBA四通道或灰度图报错）
        img = Image.open(image_path).convert('RGB')
    except FileNotFoundError:
        print(f"找不到图片：{image_path}")
        return

    # 预处理
    img_tensor = transform(img).unsqueeze(0).to(device) 
    # .unsqueeze(0) 是为了增加一个 Batch 维度，变成 [1, 3, 32, 32] 的形状

    # 使用 torch.no_grad() 关闭梯度计算，节省显存并加速
    with torch.no_grad():
        outputs = model(img_tensor)  # 获得10个类别的分数
        _, predicted = torch.max(outputs, 1)  # 获得最大分数的索引（预测类别）
        prob = torch.nn.functional.softmax(outputs, dim=1)[0] * 100 # 计算置信度概率

    pred_idx = predicted.item()
    print(f"\n图片路径: {image_path}")
    print(f"预测类别: {classes[pred_idx]} (索引: {pred_idx})")
    print(f"置信度: {prob[pred_idx].item():.2f}%")

#  运行测试
if __name__ == '__main__':

    # 定义存放图片的目录
    image_dir = './test_images'      
    # 定义图片文件名
    image_name = '_1.jpg' 
    
    # 使用 os.path.join 拼接目录和文件名，自动处理斜杠
    image_path = os.path.join(image_dir, image_name)
    
    print("开始推理...")
    predict_image(image_path)