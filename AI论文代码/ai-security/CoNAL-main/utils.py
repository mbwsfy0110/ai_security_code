import numpy as np
from torch.utils import data
import torch
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.model_selection import train_test_split
import IPython
import os
from torchvision.datasets.utils import download_url, check_integrity
import torchvision.transforms as transforms
import sys
import pandas as pd
from PIL import Image
import pickle
import torchvision.models as models
from sklearn.preprocessing import normalize


# 这段代码用于处理数据中的非连续索引，将数据映射到连续的索引范围 [0, N)，方便后续处理和分析。
def map_data(data):
    """
    Map data to proper indices in case they are not in a continues [0, N) range

    Parameters
    ----------
    data : np.int32 arrays

    Returns
    -------
    mapped_data : np.int32 arrays
    n : length of mapped_data

    """
    uniq = list(set(data))
    # 首先，将输入数据data转换为一个集合，去除重复值，然后转换为列表，得到唯一值列表    uniq
    id_dict = {old: new for new, old in enumerate(sorted(uniq))}
    data = np.array(list(map(lambda x: id_dict[x], data)))
    # 接下来，通过遍历排序后的唯一值列表uniq，创建一个字典id_dict，该字典的键是原始值，值是其在排序后的唯一值列表中的索引
    # 使用id_dict字典将输入数据data映射到适当的索引，并将其转换为NumPy数组。
    n = len(uniq)

    return data, id_dict, n


# 将给定的目标值转换为独热编码形式，适用于分类问题中对目标值的编码需求
def one_hot(target, n_classes):
    targets = np.array([target]).reshape(-1)
    one_hot_targets = np.eye(n_classes)[targets]
    return one_hot_targets


# 将answer文件中每个工人对每个样本的标注转换为独热编码，缺失值全部为-1
def transform_onehot(answers, N_ANNOT, N_CLASSES, empty=-1):
    answers_bin_missings = []
    # 初始化一个空列表，用于存储转换后的独热编码形式的目标值
    for i in range(len(answers)):  # 遍历输入的 answers 列表，其中每个元素代表一个样本
        row = []  # 存储当前样本的转换后的独热编码形式的目标值。
        for r in range(N_ANNOT):  # 遍历每个样本中的注释数 N_ANNOT，即每个样本中包含的目标值的数量。
            if answers[i, r] == -1:  # 缺失值
                row.append(empty * np.ones(N_CLASSES))  # 则将一个长度为 N_CLASSES，元素全为 empty 的数组添加到 row 列表中
            else:  # 调用one_hot函数将当前注释值转换为独热编码形式，并将结果添加到 row 列表中
                row.append(one_hot(answers[i, r], N_CLASSES)[0, :])
        answers_bin_missings.append(row)
    answers_bin_missings = np.array(answers_bin_missings).swapaxes(1, 2)
    # 并使用swapaxes函数交换数组的轴
    return answers_bin_missings


class Dataset(data.Dataset):
    def __init__(self, mode='train', k=0, dataset='labelme', sparsity=0, test_ratio=0):
        if mode[:5] == 'train':
            self.mode = mode[:5]
        else:
            self.mode = mode
        if dataset == 'music':
            data_path = '../ldmi/data/music/'
            X = np.load(data_path + 'data_%s.npy' % self.mode)
            y = np.load(data_path + 'labels_%s.npy' % self.mode).astype(np.int)
            if mode == 'train':
                answers = np.load(data_path + '/answers_15.npy')
                self.answers = answers
                self.num_users = answers.shape[1]
                classes = np.unique(answers)
                if -1 in classes:
                    self.num_classes = len(classes) - 1
                else:
                    self.num_classes = len(classes)
                self.input_dims = X.shape[1]
                self.answers_onehot = transform_onehot(answers, answers.shape[1], self.num_classes)
        if dataset == 'labelme':
            data_path = './data/LabelMe/prepared/'
            X = np.load(data_path + self.mode + '/data_%s_vgg16.npy' % self.mode)
            y = np.load(data_path + self.mode + '/labels_%s.npy' % self.mode)
            X = X.reshape(X.shape[0], -1)
            if mode == 'train':
                # answers = np.load(data_path + self.mode + '/answers_8.npy')
                answers = np.load(data_path + self.mode + '/answers_15.npy')
                # 加载工人标签数据，并保存在类中
                self.answers = answers
                self.num_users = answers.shape[1]
                classes = np.unique(answers)
                if -1 in classes:
                    self.num_classes = len(classes) - 1
                else:
                    self.num_classes = len(classes)
                self.input_dims = X.shape[1]
                self.answers_onehot = transform_onehot(answers, answers.shape[1], 8)

                # y = np.load(data_path + self.mode + '/labels_%s.npy' % self.mode)
                # y = simple_majority_voting(answers)
            elif mode == 'train_dmi':
                answers = np.load(data_path + self.mode + '/answers.npy')
                self.answers = transform_onehot(answers, answers.shape[1], 8)
                self.num_users = answers.shape[1]
                classes = np.unique(answers)
                if -1 in classes:
                    self.num_classes = len(classes) - 1
                else:
                    self.num_classes = len(classes)
                self.input_dims = X.shape[1]
        train_num = int(len(X) * (1 - test_ratio))
        self.X = torch.from_numpy(X).float()[:train_num]
        self.X_val = torch.from_numpy(X).float()[train_num:]
        # 划分验证集和训练集
        if k:#计算数据集中每个样本的前 k 个最近邻，并将结果存储在对象的属性中，以备后续使用
            dist_mat = euclidean_distances(X, X)  # 欧氏距离函数计算数据集 X 中每个样本之间的距离，并将结果存储在 dist_mat 中
            k_neighbors = np.argsort(dist_mat, 1)[:, :k]
            self.ins_feat = torch.from_numpy(X)
            self.k_neighbors = k_neighbors
            # self.X = torch.arange(0, len(dist_mat), 1)
        self.y = torch.from_numpy(y)[:train_num]
        self.y_val = torch.from_numpy(y)[train_num:]
        if mode == 'train':
            self.ans_val = answers[train_num:]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.mode == 'train':
            return idx, self.X[idx], self.answers[idx], self.answers_onehot[idx], self.y[idx]
        else:
            return idx, self.X[idx], self.y[idx]


def simple_majority_voting(response, empty=-1):
    # 定义一个简单的多数投票函数
    mv = []  # 用于存储每行的多数投票结果
    for row in response:  # 遍历每一行的投票结果
        # 统计除了空值之外的投票结果的频次
        bincount = np.bincount(row[row != empty])
        # 找到频次最高的投票结果作为该行的多数投票结果
        mv.append(np.argmax(bincount))
    # 将多数投票结果转换为 NumPy 数组并返回
    return np.array(mv)
