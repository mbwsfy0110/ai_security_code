from losses import *
import torch
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import f1_score
import IPython

loss_fn = torch.nn.CrossEntropyLoss(reduction='mean').to(
    "cuda" if torch.cuda.is_available() else 'cpu')  # 创建一个使用均值归约的交叉熵损失函数


def train(train_loader, model, optimizer, criterion=F.cross_entropy, mode='simple', annotators=None, pretrain=None,
          support=None, support_t=None, scale=0):
    model.train()  # 模型设置为训练模式b
    correct = 0
    total = 0
    total_loss = 0
    loss = 0

    correct_rec = 0
    total_rec = 0
    for idx, input, targets, targets_onehot, true_labels in train_loader:
        input = input.to("cuda" if torch.cuda.is_available() else 'cpu')
        targets = targets.to("cuda" if torch.cuda.is_available() else 'cpu').long()
        targets_onehot = targets_onehot.to("cuda" if torch.cuda.is_available() else 'cpu')
        targets_onehot[targets_onehot == -1] = 0
        # 将 targets_onehot 中的值为 -1 的元素替换为 0
        true_labels = true_labels.to("cuda" if torch.cuda.is_available() else 'cpu').long()

        if mode == 'simple':  # 准确率最差
            loss = 0
            if scale:
                cls_out, output, trace_norm = model(input)
                # cls_out,加权输出的结果，模型的迹范数
                loss += scale * trace_norm
                mask = targets != -1
                y_pred = torch.transpose(output, 1, 2)
                y_true = torch.transpose(targets_onehot, 1, 2).float()
                # 使用交叉熵损失计算分类任务的损失值
                loss += torch.mean(-y_true[mask] * torch.log(y_pred[mask]))
            else:
                cls_out, output = model(input)
                loss += criterion(targets, output)  # F定义的交叉熵损失
            _, predicted = cls_out.max(1)
            correct += predicted.eq(true_labels).sum().item()
            total += true_labels.size(0)
        elif mode == 'common':  # 准确率其次
            rec_loss = 0
            loss = 0
            cls_out, output = model(input, mode='train')
            _, predicted = cls_out.max(1)
            correct += predicted.eq(true_labels).sum().item()
            total += true_labels.size(0)
            loss += criterion(targets, output)  # F定义的交叉熵损失
            loss -= 0.00001 * torch.sum(
                torch.norm((model.kernel - model.common_kernel).view(targets.shape[1], -1), dim=1, p=2))
        # 对损失减去额外的惩罚项，这个惩罚项是模型model的kernel与common_kernel
        # 之间的欧氏距离，乘以一个小的系数0.00001
        elif mode=='xiaorong':#消融实验
            rec_loss = 0
            loss = 0
            cls_out, output = model(input, mode='train')
            _, predicted = cls_out.max(1)
            correct += predicted.eq(true_labels).sum().item()
            total += true_labels.size(0)
            loss += criterion(targets, cls_out.unsqueeze(2).expand(-1, -1, 59))  # F定义的交叉熵损失
            loss -= 0.00001 * torch.sum(
                torch.norm((model.kernel - model.common_kernel).view(targets.shape[1], -1), dim=1, p=2))
        else:  # 准确率最高
            output, _ = model(input)
            loss = loss_fn(output, true_labels)  # 只使用crowd_out和交叉熵损失计算损失
            _, predicted = output.max(1) #预测的结果是加权之后的
            correct += predicted.eq(true_labels).sum().item()
            total += true_labels.size(0)
        total_loss += loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # if mode == 'simple' or mode == 'common':
        # print('Training acc: ', correct / total)
        return correct / total


def test(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    target = []
    predict = []
    for _, inputs, targets in test_loader:
        inputs = inputs.to("cuda" if torch.cuda.is_available() else 'cpu')
        target.extend(targets.data.numpy())
        targets = targets.to("cuda" if torch.cuda.is_available() else 'cpu')

        total += targets.size(0)
        output, _ = model(inputs, mode='test')
        _, predicted = output.max(1)
        predict.extend(predicted.cpu().data.numpy())
        correct += predicted.eq(targets).sum().item()
    acc = correct / total
    f1 = f1_score(target, predict, average='macro')

    classes = list(set(target))
    classes.sort()
    acc_per_class = []  # 存储每个类别的准确率
    predict = np.array(predict)
    target = np.array(target)
    for i in range(len(classes)):
        instance_class = target == i
        acc_i = np.mean(predict[instance_class] == classes[i])
        acc_per_class.append(acc_i)
    return acc, f1, acc_per_class  # 返回分类准确率、F1分数和每个类别的准确率
