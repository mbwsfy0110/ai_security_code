import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

# 加载预训练的VGG16模型
vgg16 = models.vgg16(pretrained=True)

# 移除VGG16模型的全连接层（分类器部分）
vgg16_features = list(vgg16.features.children())[:-1]
vgg16 = torch.nn.Sequential(*vgg16_features)

# 定义数据预处理和加载
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 加载图像数据集
dataset = ImageFolder(root='your_dataset_path', transform=transform)

# 创建数据加载器
data_loader = DataLoader(dataset, batch_size=20, shuffle=False)

# 提取特征
features = []
for images, _ in data_loader:
    # 将图像数据输入到VGG16模型中，并获取特征
    with torch.no_grad():
        outputs = vgg16(images)
    features.append(outputs)

# 将特征堆叠为一个张量
features_tensor = torch.cat(features, dim=0)

# 打印特征的形状
print(features_tensor.shape)  # 应该输出 [20, 512, 4, 4]
