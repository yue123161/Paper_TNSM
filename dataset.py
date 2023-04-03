from collections import Counter, Callable

import numpy as np
import torch
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from torch.utils.data import Sampler, Dataset

class MyDataset(Dataset):
    def __init__(self,features,labels):
        self.x_data=features
        self.y_data=labels
        self.len=len(labels)

    def __getitem__(self, index):
        return self.x_data[index],self.y_data[index]

    def __len__(self):
        return self.len

class MyDatasetSampler(Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset
    从不均衡数据中进行采样
    Arguments:
        indices: a list of indices
        num_samples: number of samples to draw
        callback_get_label: a callback-like function which takes two arguments - dataset and index
    """
    def __init__(self, dataset, indices: list = None, num_samples: int = None, callback_get_label: Callable = None):
        # if indices is not provided, all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) if indices is None else indices

        # define custom callback
        self.callback_get_label = callback_get_label

        # if num_samples is not provided, draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) if num_samples is None else num_samples

        # distribution of classes in the dataset
        label_to_count = {}
        for idx in self.indices:
            label = self._get_label(dataset, idx)
            label_to_count[label] = label_to_count.get(label, 0) + 1

        # weight for each sample
        # 求每个样本的权重
        weights = [1.0 / label_to_count[self._get_label(dataset, idx)] for idx in self.indices]
        self.weights = torch.DoubleTensor(weights)

    def _get_label(self, dataset, idx):
        if self.callback_get_label:
            return self.callback_get_label(dataset, idx)
        elif isinstance(dataset,MyDataset):
            return dataset.y_data[idx]
        else:
            raise NotImplementedError

    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples

def data_reader(file_path):
    """
    从路径进行数据读取
    :param file_path:文件路径
    :return:
    """
    data=np.load(file_path)
    feature=data[:,0:-1]
    label=data[:,-1]
    return feature,label

def resampler(feature,label):
    """
    针对数据不均衡问题，对数据进行重采样
    采用SMOTE
    :param feature:
    :param label:
    :return:
    """
    over = RandomOverSampler()
    under = RandomUnderSampler()
    steps = [('o', over), ('u', under)]
    pipeline = Pipeline(steps=steps)

    feature, label = pipeline.fit_resample(feature, label)
    print(Counter(label))

    return feature,label

def get_k_fold_data(k, i, X, y):  ###此过程主要是步骤（1）
    # 返回第i折交叉验证时所需要的训练和验证数据，分开放，X_train为训练数据，X_valid为验证数据
    assert k > 1
    fold_size = X.shape[0] // k  # 每份的个数:数据总条数/折数（组数）

    X_train, y_train = None, None
    X_valid,y_valid=None,None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)  # slice(start,end,step)切片函数
        ##idx 为每组 valid
        X_part, y_part = X[idx, :], y[idx]
        if j == i:  ###第i折作valid
            X_valid, y_valid = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = np.concatenate([X_train, X_part], axis=0)  # axis=0增加行数，竖着连接
            y_train = np.concatenate([y_train, y_part], axis=0)
    return X_train, y_train, X_valid, y_valid


class MyDataset_idx(Dataset):
    def __init__(self,features,labels):
        self.x_data=features
        self.y_data=labels
        self.len=len(labels)

    def __getitem__(self, index):
        return self.x_data[index],self.y_data[index],index

    def __len__(self):
        return self.len



class Dataset_Multi(Dataset):
    def __init__(self,features,labels):
        self.x_data=features
        self.y_data=labels
        self.len=len(labels)

    def __getitem__(self, index):
        return self.x_data[index],self.y_data[index]

    def __len__(self):
        return self.len