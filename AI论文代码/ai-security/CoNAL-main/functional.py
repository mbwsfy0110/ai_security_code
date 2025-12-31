# 导入需要的库
from torch.distributions import RelaxedOneHotCategorical, RelaxedBernoulli  # 导入 PyTorch 的分布模块
import torch.nn as nn  # 导入 PyTorch 的神经网络模块


# 定义一个函数，实现 Gumbel Sigmoid 操作
def gumbel_sigmoid(input, temp):
    # 使用 Gumbel 分布生成 Sigmoid 近似采样
    return RelaxedBernoulli(temp, probs=input).rsample()


# 定义一个继承自 nn.Module 的类，实现 Gumbel Sigmoid 操作的模块化
class GumbelSigmoid(nn.Module):
    def __init__(self,
                 temp: float = 0.1,  # 初始化函数，指定温度参数，默认为0.1
                 threshold: float = 0.5):  # 初始化函数，指定阈值参数，默认为0.5
        super(GumbelSigmoid, self).__init__()  # 调用父类的初始化函数
        self.temp = temp  # 存储临时参数
        self.threshold = threshold  # 存储阈值参数

    def forward(self, input):
        if self.training:  # 如果处于训练模式
            # 使用 Gumbel Sigmoid 函数进行近似采样
            return gumbel_sigmoid(input, self.temp)
        else:  # 如果处于推理模式
            # 将输入通过 Sigmoid 函数，并根据阈值进行二值化处理
            return (input.sigmoid() >= self.threshold).float()
