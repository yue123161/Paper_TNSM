"""
   比对损失实现几个关键模块
   1、损失函数，已经实现
   2、编码模块
   3、映射网络
   4、数据增强模块
   5、联合损失函数
   6、温度调试
"""
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, \
    classification_report
from torch import nn
import torch
from torch.nn import functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm


from dataset import MyDataset, MyDatasetSampler
from dataset import data_reader
import numpy as np

from matplotlib import pyplot as plt

seed = 32
torch.manual_seed(seed)

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

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
            nn.Linear(196, 256),
            nn.ReLU(),
        )
        self.l2 = nn.Sequential(
            nn.Linear(256, 128,bias=False),
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
        #x = F.normalize(x, dim=1)  # 对输入进行正则化
        feature = F.normalize(self.head(x), dim=1)
        return feature


#####
### 这里需要考虑需不需要对x进行normalize
#####
# 定义分类器网络
class LinearClassifier(nn.Module):
    def __init__(self):
        super(LinearClassifier, self).__init__()
        self.fc = nn.Linear(128, 2)

    def forward(self, feature):
        # feature=F.normalize(feature)
        return self.fc(feature)


def mixup_data(x, y, lamda=0.95):
    batch_size = x.size()[0]
    index = torch.randperm(batch_size)  # 随机做mixup
    mixed_x_a = lamda * x + (1 - lamda) * x[index, :]
    mixed_x_b = lamda * x[index, :] + (1 - lamda) * x
    y_a, y_b = y, y[index]
    return mixed_x_a, y_a, mixed_x_b, y_b




"""
   比对损失函数不需要修改
"""

if __name__ == '__main__':
    train_path = './data/unsw_fine/train.npy'
    test_path = './data/unsw_fine/test.npy'

    train_X, train_Y = data_reader(train_path)

    test_X, test_Y = data_reader(test_path)

    print(set(test_Y))

    encoder = Encoder().to(device)
    header = Header(header='linear').to(device)
    classifier = LinearClassifier().to(device)

    lr = 1e-5  # 学习率

    num_epoches = 60  # 每则数据20次

    batch_size = 128

    criterion_sup = SupConLoss(temperature=0.07)
    criterion_ce = nn.CrossEntropyLoss(reduction='none')
    optimizer = Adam(
        [
            {'params': encoder.parameters(), 'lr': lr},  # 这里失误，
            {'params': header.parameters(), 'lr': lr},
            {'params': classifier.parameters(), 'lr': lr}
        ]
    )

    history_train_loss_mul = []
    history_train_acc_mul = []

    history_test_loss_mul = []
    history_test_acc_mul = []

    # 进行k_fold训练

    test_dataset = MyDataset(test_X, test_Y)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    train_dataset = MyDataset(train_X, train_Y)

    sampler = MyDatasetSampler(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)

    #aucroc = torchmetrics.AUROC(num_classes=2, pos_label=0)

    split_index = 37

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
            # 1 连续数据增加高斯噪声
            data_noise = torch.randn(data.size()).to(device)* 0.0001 + data

            data_noise_1 = torch.randn(data.size()).to(device)* 0.00025 + data

            data_noise_2 = torch.randn(data.size()).to(device) * 0.0005 + data

            # 2 连续数据做mixup

            data_mixup, label_mixup, _, _ = mixup_data(data, label, 0.9999)

            data_mixup_1, label_mixup_1, _, _ = mixup_data(data, label, 0.99975)

            data_mixup_2, label_mixup_2, _, _ = mixup_data(data, label, 0.9995)

            data_new = torch.cat(
                [data, data_noise, data_noise_1, data_noise_2, data_mixup, data_mixup_1, data_mixup_2], dim=0)

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

            # print(str(loss_ce)+" "+str(loss_sup))

            loss.backward()

            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.5)
            torch.nn.utils.clip_grad_norm_(header.parameters(), 1.5)
            torch.nn.utils.clip_grad_norm_(classifier.parameters(), 1.5)

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
        history_train_loss_mul.append(train_loss / train_total)
        history_train_acc_mul.append(train_correct / train_total)

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

            # aucroc(output,label)

            test_bar.set_description(
                'Test Epoch: [{}/{}] Loss: {:.4f} ACC:{:.4f} '.format(epoch, num_epoches,
                                                                      test_loss / test_total,
                                                                      test_correct / test_total))
        history_test_loss_mul.append(test_loss / test_total)
        history_test_acc_mul.append(test_correct / test_total)

        # total_auc=aucroc.compute()

        # print(total_auc)

        #aucroc.reset()





    # 计算误报、漏报情况
    # 原始训练、测试标签
    # 定义原始数据准确率、查全率

    label_test = []
    label_predict_test = []
    score_predict_test = []
    test_bar = tqdm(test_loader)
    for data, label in test_bar:
        label = label != 0
        label=label.long()
        label_test.append(label.numpy())
        label=label.to(device)
        data = data.float().to(device)
        feature = encoder(data)
        output = classifier(feature)

        predict = torch.max(output, 1)[1]
        label_predict_test.append(predict.numpy())
        score_predict_test.append(output.detach().numpy())


    label_test = np.concatenate(label_test)
    label_predict_test = np.concatenate(label_predict_test)
    score_predict_test = np.concatenate(score_predict_test)

    cm = confusion_matrix(label_test, label_predict_test)
    np.set_printoptions(precision=2)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure()

    plt.imshow(cm_normalized, cmap=plt.cm.Blues)

    plt.title('Test Confusion Matrix')

    indices = range(len(cm_normalized))

    plt.xticks(indices, ['normal', 'malware'])  # 'dos', 'u2r', 'r2l', 'probe'])

    plt.yticks(indices, ['normal', 'malware'])  # 'dos', 'u2r', 'r2l', 'probe'])

    plt.colorbar()

    plt.xlabel('Predict Label')

    plt.ylabel('True Label')

    for first_index in range(len(cm_normalized)):  # 第几行
        for second_index in range(len(cm_normalized[first_index])):  # 第几列
            plt.text(second_index, first_index, "%0.2f" % cm_normalized[first_index][second_index], color='red')
    plt.show()

    # 计算准确率、召回率、误报率

    test_acc = accuracy_score(label_test, label_predict_test)
    test_precision = precision_score(label_test, label_predict_test, average='macro')
    test_recall = recall_score(label_test, label_predict_test, average='macro')
    test_f1 = f1_score(label_test, label_predict_test, average='macro')
    print("测试集准确率为:" + str(test_acc))
    print("测试集精确率为:" + str(test_precision))
    print("测试集recall为:" + str(test_recall))
    print("测试集f1为:" + str(test_f1))

    ans = classification_report(label_test, label_predict_test)
    print(ans)

    fpr = (float(cm[0, 1]) / float((cm[0, 1] + cm[0, 0])))

    print("fpr score:" + str(fpr))
    print("\n")


# 测试集准确率为:0.9301646505951261
# 测试集精确率为:0.9231104391329916
# 测试集recall为:0.9150796985409158
# 测试集f1为:0.9189255798382632
#               precision    recall  f1-score   support
#
#            0       0.90      0.87      0.89     56000
#            1       0.94      0.96      0.95    119341
#
#     accuracy                           0.93    175341
#    macro avg       0.92      0.92      0.92    175341
# weighted avg       0.93      0.93      0.93    175341













