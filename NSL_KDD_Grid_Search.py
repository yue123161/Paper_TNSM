"""
   比对损失实现几个关键模块
   1、损失函数，已经实现
   2、编码模块
   3、映射网络
   4、数据增强模块
   5、联合损失函数
   6、温度调试
"""

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, \
    classification_report
from torch import nn
import torch
from torch.nn import functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

from dataset import MyDataset, MyDatasetSampler
from dataset import data_reader
import numpy as np

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""

    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]  # 定义contrast的数量
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).sum(0)
        return loss


# 定义编码网络
# 确定不需要dropout
# 目前不确定需不需要去掉relu
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.l1 = nn.Sequential(
            nn.Linear(121, 256),
            nn.ReLU(),
        )
        self.l2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        self.l3 = nn.Sequential(
            nn.Linear(128, 128, bias=False)
        )

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        return x


class Header(nn.Module):
    def __init__(self, header='linear'):
        super(Header, self).__init__()
        if header == 'linear':
            self.head = nn.Linear(128, 128, bias=False)
        elif header == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(128, 256),
                nn.ReLU(),
                nn.Linear(256, 128)
            )
        else:
            raise NotImplementedError(
                'head not supported: {}'.format(header))

    def forward(self, x):
        x = F.normalize(x, dim=-1)  # 对输入进行正则化
        feature = F.normalize(self.head(x), dim=-1)
        return feature



class LinearClassifier(nn.Module):
    def __init__(self):
        super(LinearClassifier, self).__init__()
        self.fc = nn.Linear(128, 2)

    def forward(self, feature):
        return self.fc(feature)


def mixup_data(x, y, lamda=0.95):
    batch_size = x.size()[0]
    index = torch.randperm(batch_size)  # 随机做mixup
    mixed_x_a = lamda * x + (1 - lamda) * x[index, :]
    mixed_x_b = lamda * x[index, :] + (1 - lamda) * x
    y_a, y_b = y, y[index]
    return mixed_x_a, y_a, mixed_x_b, y_b


import pandas as pd
def data_write_csv(file_name, datas):  # file_name为写入CSV文件的路径，datas为要写入数据列表
    df=pd.DataFrame(datas,columns=['step','train Loss','train accuracy','test loss','test accuracy'])
    df.to_csv(file_name)
    print("模型训练结束")

"""
   比对损失函数不需要修改

"""

def main(temperature,mask_K,result_file):
    """
    :param temperature: 温度
    :param mask_K: 遮罩数量
    :param result_file: 结果文件
    :return:
    """

    seed = 32
    torch.manual_seed(seed)

    train_path = './data/nsl_kdd/train.npy'
    test_path = './data/nsl_kdd/test.npy'

    train_X, train_Y = data_reader(train_path)

    test_X, test_Y = data_reader(test_path)

    encoder = Encoder().to(device)
    header = Header(header='linear').to(device)
    classifier = LinearClassifier().to(device)

    lr = 1e-4  # 学习率

    num_epoches = 20  # 每则数据20次

    batch_size = 128

    temperature=temperature

    criterion_sup = SupConLoss(temperature=temperature)
    criterion_ce = nn.CrossEntropyLoss(reduction='none')
    optimizer = Adam(
        [
            {'params': encoder.parameters(), 'lr': lr},  # 这里失误，
            {'params': header.parameters(), 'lr': lr},
            {'params': classifier.parameters(), 'lr': lr}
        ]
    )


    # 进行训练
    datas = []

    test_dataset = MyDataset(test_X, test_Y)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    train_dataset = MyDataset(train_X, train_Y)

    sampler = MyDatasetSampler(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)

    K=37
    input_lenth=121

    for epoch in range(num_epoches):
        train_loss, train_total, train_correct, train_bar = 0.0, 0, 0, tqdm(train_loader)
        encoder.train()
        header.train()
        classifier.train()

        for data, label in train_bar:
            optimizer.zero_grad()
            data = data.float().to(device)
            label = label != 0
            label = label.long().to(device)
            # 数据操作
            data_0 = data.clone()
            mask_temp=torch.randint(high=K,size=(len(label),mask_K))
            mask_temp_one_hot=(F.one_hot(mask_temp,num_classes=input_lenth)==0)
            mask_0=mask_temp_one_hot[:,0,:].int().float()
            if(mask_temp_one_hot.size(1)>=2):
                for i in range(1,mask_temp_one_hot.size(1)):
                    mask_0=mask_0*(mask_temp_one_hot[:, i, :].int().float())
            mask_0=mask_0.to(device)
            data_0=data_0*mask_0

            data_1 = data.clone()
            mask_temp = torch.randint(high=K, size=(len(label), mask_K))
            mask_temp_one_hot = (F.one_hot(mask_temp, num_classes=input_lenth) == 0)
            mask_1 = mask_temp_one_hot[:, 0, :].int().float()
            if (mask_temp_one_hot.size(1) >= 2):
                for i in range(1, mask_temp_one_hot.size(1)):
                    mask_1 = mask_1 * (mask_temp_one_hot[:, i, :].int().float())
            mask_1 = mask_1.to(device)
            data_1= data_1 * mask_1

            data_2 = data.clone()
            mask_temp = torch.randint(high=K, size=(len(label), mask_K))
            mask_temp_one_hot = (F.one_hot(mask_temp, num_classes=input_lenth) == 0)
            mask_2 = mask_temp_one_hot[:, 0, :].int().float()
            if (mask_temp_one_hot.size(1) >= 2):
                for i in range(1, mask_temp_one_hot.size(1)):
                    mask_2 = mask_2* (mask_temp_one_hot[:, i, :].int().float())
            mask_2 = mask_2.to(device)
            data_2= data_2 * mask_2

            data_3 = data.clone()
            mask_temp = torch.randint(high=K, size=(len(label), mask_K))
            mask_temp_one_hot = (F.one_hot(mask_temp, num_classes=input_lenth) == 0)
            mask_3 = mask_temp_one_hot[:, 0, :].int().float()
            if (mask_temp_one_hot.size(1) >= 2):
                for i in range(1, mask_temp_one_hot.size(1)):
                    mask_3 = mask_3 * (mask_temp_one_hot[:, i, :].int().float())
            mask_3 = mask_3.to(device)
            data_3 = data_3 * mask_3

            data_4 = data.clone()
            mask_temp = torch.randint(high=K, size=(len(label), mask_K))
            mask_temp_one_hot = (F.one_hot(mask_temp, num_classes=input_lenth) == 0)
            mask_4 = mask_temp_one_hot[:, 0, :].int().float()
            if (mask_temp_one_hot.size(1) >= 2):
                for i in range(1, mask_temp_one_hot.size(1)):
                    mask_4 = mask_4 * (mask_temp_one_hot[:, i, :].int().float())
            mask_4 = mask_4.to(device)
            data_4 = data_4 * mask_4

            data_5 = data.clone()
            mask_temp = torch.randint(high=K, size=(len(label), mask_K))
            mask_temp_one_hot = (F.one_hot(mask_temp, num_classes=input_lenth) == 0)
            mask_5= mask_temp_one_hot[:, 0, :].int().float()
            if (mask_temp_one_hot.size(1) >= 2):
                for i in range(1, mask_temp_one_hot.size(1)):
                    mask_5 = mask_5 * (mask_temp_one_hot[:, i, :].int().float())
            mask_5 = mask_5.to(device)
            data_5 = data_5 * mask_5

            data_new = torch.cat(
                [data, data_0, data_1, data_2, data_3, data_4, data_5], dim=0)

            bsz = label.shape[0]

            features = encoder(data_new)

            features = header(features)

            f1, f2, f3, f4, f5, f6, f7 = torch.split(features, [bsz, bsz, bsz, bsz, bsz, bsz, bsz], dim=0)

            features = torch.cat(
                [f1.unsqueeze(1), f2.unsqueeze(1), f3.unsqueeze(1), f4.unsqueeze(1), f5.unsqueeze(1),
                 f6.unsqueeze(1), f7.unsqueeze(1)], dim=1)

            loss_sup = criterion_sup(features, label)

            feature = encoder(data)

            output = classifier(feature)

            loss_ce = criterion_ce(output, label)

            loss = (loss_sup * loss_ce)

            loss = loss.sum()
            loss.backward()


            optimizer.step()

            train_loss += loss.item()
            train_total += len(label)
            predict = torch.max(output, 1)[1]
            train_correct += (predict == label).sum()
            train_bar.set_description(
                'Train Epoch: [{}/{}] Loss_Sup: {:.4f}  Loss_Ce:{:.4f} ACC:{:.4f}'.format(epoch, num_epoches,
                                                                                          loss_sup.mean(),
                                                                                          loss_ce.mean(),
                                                                                          train_correct / train_total))


        test_loss, test_total, test_correct, test_bar = 0.0, 0, 0, tqdm(test_loader)
        encoder.eval()
        header.eval()
        classifier.eval()

        for data, label in test_bar:
            data = data.float().to(device)
            label = label != 0
            label = label.long().to(device)

            feature = encoder(data)

            output = classifier(feature)

            loss = criterion_ce(output, label).mean()

            test_loss += loss.item()
            test_total += len(label)

            predict = torch.max(output, 1)[1]

            test_correct += (predict == label).sum()


            test_bar.set_description(
                'Test Epoch: [{}/{}] Loss: {:.4f} ACC:{:.4f} '.format(epoch, num_epoches,
                                                                      test_loss / test_total,
                                                                      test_correct / test_total))

        datas.append([epoch, train_loss / train_total, train_correct / train_total, test_loss / test_total,
                      test_correct / test_total])





    label_test = []
    label_predict_test = []
    test_bar = tqdm(test_loader)
    for data, label in test_bar:
        label = label != 0
        label=label.long().to(device)
        label_test.append(label.cpu().numpy())
        data = data.float().to(device)
        feature = encoder(data)
        output = classifier(feature)

        predict = torch.max(output, 1)[1]
        label_predict_test.append(predict.cpu().numpy())


    label_test = np.concatenate(label_test)
    label_predict_test = np.concatenate(label_predict_test)

    title = "dnn_contrast-" + str(temperature) + "-" + str(mask_K)



    # 计算准确率、召回率、误报率
    print("xxxxxxxxxxxxxxxxxxxxxxx  Start xxxxxxxxxxxxxxxxxxxxx", file=result_file)
    print("--------------------------------------------------", file=result_file)
    print("Temperature:" + str(temperature), file=result_file)
    print("Mask number:" + str(mask_K), file=result_file)
    test_acc = accuracy_score(label_test, label_predict_test)
    test_precision = precision_score(label_test, label_predict_test, average='macro')
    test_recall = recall_score(label_test, label_predict_test, average='macro')
    test_f1 = f1_score(label_test, label_predict_test, average='macro')
    print("Accuracy:" + str(test_acc), file=result_file)
    print("Precision:" + str(test_precision), file=result_file)
    print("Recall:" + str(test_recall), file=result_file)
    print("f1 score:" + str(test_f1), file=result_file)

    ans = classification_report(label_test, label_predict_test)
    print(ans, file=result_file)

    print("--------------------------------------------------", file=result_file)
    print("xxxxxxxxxxxxxxxxxxxxxxx  end xxxxxxxxxxxxxxxxxxxxx", file=result_file)

    data_write_csv('./results/nslkdd_dnn_model_20211109' + title + '.csv', datas)


if __name__ == '__main__':
    temperatures = [
        1.0,
        0.6,
        0.4,
        0.2,
        0.07,
        0.06,
        0.05,
        0.04,
        0.01,
        0.007,
        0.005
    ]
    mask_Ks = [9,7,5,3,2,1]

    for temperature in temperatures:
        for mask_K in mask_Ks:
            result_file = open("nslkdd_dnn_contrast_20211109.txt", "a")
            main(temperature, mask_K, result_file)
            result_file.close()













