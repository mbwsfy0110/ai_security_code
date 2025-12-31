import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from PIL import Image
from torch import nn


# 定义修改后的VGG16模型
class ModifiedVGG16(nn.Module):
    def __init__(self):
        super(ModifiedVGG16, self).__init__()
        self.features = models.vgg16(pretrained=True).features
        self.avgpool = nn.AdaptiveAvgPool2d((4, 4))
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        return x


# 加载预训练的VGG16模型
vgg16 = models.vgg16(pretrained=True)

# 加载数据集图像文件
mean_ = np.load("./testResult/process/mean.npy")
std_ = np.load("./testResult/process/variance.npy")

# 定义数据预处理和加载
transform = transforms.Compose([
    transforms.Resize(256),#128
    transforms.ToTensor(),
    transforms.Normalize(mean=mean_, std=std_)
])

# 加载图像数据集
dataset = ImageFolder(root='./testData', transform=transform)
# 获取数据集大小
num_samples = len(dataset)
# 初始化空数组来保存标签
labels = np.zeros(num_samples)
# 遍历数据集并获取标签
for i in range(num_samples):
    # 获取第i个样本的标签
    labels[i] = dataset[i][1]
# 创建数据加载器
data_loader = DataLoader(dataset, batch_size=12, shuffle=False)
# 初始化修改后的VGG16模型
vgg16_modified = ModifiedVGG16()
# 设置模型为评估模式
vgg16_modified.eval()

# 提取特征并记录图像路径和特征索引
features = torch.zeros(len(dataset), 512, 4, 4)
image_paths = []
bat_size=data_loader.batch_size
for i, (images, _) in enumerate(data_loader):
    # 记录每个样本的起始路径和索引
    # if i>0:break
    start_index = i * data_loader.batch_size
    image_paths.extend(dataset.imgs[start_index:start_index + len(images)])

    # # 将图像数据输入到VGG16模型中，并获取特征
    with torch.no_grad():
        outputs = vgg16_modified(images)
    features[i*bat_size:i*bat_size+len(images)] = outputs.view(len(images), 512, 4, 4)
    # with torch.no_grad():
    #     outputs = vgg16.features(images)
    #     print(outputs.shape)
    # features[i*bat_size:i*bat_size+len(images)] = outputs.view(len(images), 512, 4, 4)

# 将特征堆叠为一个张量,并转换特征维度
features_tensor = features.permute(0, 3, 2, 1)

# 选择其中一个特征输出并打印对应的原始图像
selected_feature_index = 0  # 选择第一个特征
selected_feature = features_tensor[selected_feature_index]
selected_image_path = image_paths[selected_feature_index][0]  # 获取对应的图像路径

# 打印对应的原始图像
selected_image = Image.open(selected_image_path)
# selected_image.show()  # 显示图像
# 打印特征的形状
np.save("data/LabelMe/prepared/test_sys/data_test_sys_vgg16.npy", features_tensor)
np.save("data/LabelMe/prepared/test_sys/labels_test_sys.npy", labels)
np.save("./data/LabelMe/prepared/test_sys/image_paths.npy", image_paths)
print(features_tensor.shape)  # 应该输出 [*, 4, 4, 512]
print(labels.shape)