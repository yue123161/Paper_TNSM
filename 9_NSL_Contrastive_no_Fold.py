import copy

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import torch
from sklearn.model_selection import train_test_split
from torch import nn, optim
from torch.nn import functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

import numpy as np

from aux_util import plot_reliability_diagrams, plot_histograms
from dataset import data_reader, MyDataset, MyDatasetSampler
from model.CNN1D_Contrastive_N import Encoder as Encoder_CNN1D
from model.CNN1D_Contrastive_N import Header as Header_CNN1D
from model.CNN1D_Contrastive_N import LinearClassifier as Classifier_CNN1D

from model.DNN_Contrastive_N import Encoder as Encoder_DNN
from model.DNN_Contrastive_N import Header as Header_DNN
from model.DNN_Contrastive_N import LinearClassifier as Classifier_DNN

from model.CNN_Contrastive_N import Encoder as Encoder_CNN
from model.CNN_Contrastive_N import Header as Header_CNN
from model.CNN_Contrastive_N import LinearClassifier as Classifier_CNN





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
        # device = (torch.device('cuda')
        #           if features.is_cuda
        #           else torch.device('cpu'))

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



if __name__ == '__main__':
    print("读取数据")
    train_path = './data/nsl_kdd/train.npy'
    test_path = './data/nsl_kdd/test.npy'

    train_X, train_Y = data_reader(train_path)
    print("训练数据读取完成")
    test_X, test_Y = data_reader(test_path)
    print("测试数据读取完毕")

    batch_size=128

    test_dataset = MyDataset(test_X, test_Y)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # 对训练数据进行切分，将其分为训练集和验证集
    # train_X, val_X, train_Y, val_Y = train_test_split(train_X, train_Y, test_size=0.3, shuffle=True)
    train_dataset = MyDataset(train_X, train_Y)
    train_sampler = MyDatasetSampler(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)

    # val_dataset = MyDataset(val_X, val_Y)
    # val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    num_classes=2
    encoder_dnn=Encoder_DNN().to(device)
    encoder_cnn = Encoder_CNN().to(device)
    encoder_cnn1d=Encoder_CNN1D().to(device)
    encoders=[encoder_dnn,encoder_cnn,encoder_cnn1d]

    header_dnn=Header_DNN().to(device)
    header_cnn=Header_CNN().to(device)
    header_cnn1d=Header_CNN1D().to(device)

    headers=[header_dnn,header_cnn,header_cnn1d]

    classifier_dnn=Classifier_DNN().to(device)
    classifier_cnn=Classifier_CNN().to(device)
    classifier_cnn1d=Classifier_CNN1D().to(device)

    classifiers=[classifier_dnn,classifier_cnn,classifier_cnn1d]
    model_names=['dnn','cnn','cnn1d']

    lr=1e-4
    num_epoches = 20

    data_name='nsl_contrastive'

    for i in range(len(model_names)):
        encoder = encoders[i]
        header=headers[i]
        classifier=classifiers[i]

        model_name=model_names[i]

        print(model_name)

        eta = 1e-8

        temperature = 0.2

        criterion_sup = SupConLoss(temperature=temperature)
        criterion_ce = nn.CrossEntropyLoss(reduction='none')
        optimizer = Adam(
            [
                {'params': encoder.parameters(), 'lr': lr},  # 这里失误，
                {'params': header.parameters(), 'lr': lr},
                {'params': classifier.parameters(), 'lr': lr}
            ]
        )


        #Adam(list(model.parameters()), lr=lr)
        result_file = open("./results/9_" + str(data_name) + "_" + str(model_name) + ".txt", "a")


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

                numeric_length=37
                input_length=121
                mask_K=3

                mask_temp = torch.randint(high=numeric_length, size=(len(label), mask_K))
                mask_temp_one_hot = (F.one_hot(mask_temp, num_classes=input_length) == 0)
                mask_0 = mask_temp_one_hot[:, 0, :].int().float()
                if (mask_temp_one_hot.size(1) >= 2):
                    for i in range(1, mask_temp_one_hot.size(1)):
                        mask_0 = mask_0 * (mask_temp_one_hot[:, i, :].int().float())
                mask_0 = mask_0.to(device)
                data_0 = data * mask_0

                # 2 连续数据增加高斯噪声
                mask_temp = torch.randint(high=numeric_length, size=(len(label), mask_K))
                mask_temp_one_hot = (F.one_hot(mask_temp, num_classes=input_length) == 0)
                mask_1 = mask_temp_one_hot[:, 0, :].int().float()
                if (mask_temp_one_hot.size(1) >= 2):
                    for i in range(1, mask_temp_one_hot.size(1)):
                        mask_1 = mask_1 * (mask_temp_one_hot[:, i, :].int().float())
                mask_1 = mask_1.to(device)
                data_1 = data * mask_1

                # 3 连续数据增加高斯噪声
                mask_temp = torch.randint(high=numeric_length, size=(len(label), mask_K))
                mask_temp_one_hot = (F.one_hot(mask_temp, num_classes=input_length) == 0)
                mask_2 = mask_temp_one_hot[:, 0, :].int().float()
                if (mask_temp_one_hot.size(1) >= 2):
                    for i in range(1, mask_temp_one_hot.size(1)):
                        mask_2 = mask_2 * (mask_temp_one_hot[:, i, :].int().float())
                mask_2 = mask_2.to(device)
                data_2 = data * mask_2

                # 4 连续数据增加高斯噪声
                mask_temp = torch.randint(high=numeric_length, size=(len(label), mask_K))
                mask_temp_one_hot = (F.one_hot(mask_temp, num_classes=input_length) == 0)
                mask_3 = mask_temp_one_hot[:, 0, :].int().float()
                if (mask_temp_one_hot.size(1) >= 2):
                    for i in range(1, mask_temp_one_hot.size(1)):
                        mask_3 = mask_3 * (mask_temp_one_hot[:, i, :].int().float())
                mask_3 = mask_3.to(device)
                data_3 = data * mask_3

                # 5 连续数据增加高斯噪声
                mask_temp = torch.randint(high=numeric_length, size=(len(label), mask_K))
                mask_temp_one_hot = (F.one_hot(mask_temp, num_classes=input_length) == 0)
                mask_4 = mask_temp_one_hot[:, 0, :].int().float()
                if (mask_temp_one_hot.size(1) >= 2):
                    for i in range(1, mask_temp_one_hot.size(1)):
                        mask_4 = mask_4 * (mask_temp_one_hot[:, i, :].int().float())
                mask_4 = mask_4.to(device)
                data_4 = data * mask_4

                # 6 连续数据增加高斯噪声
                mask_temp = torch.randint(high=numeric_length, size=(len(label), mask_K))
                mask_temp_one_hot = (F.one_hot(mask_temp, num_classes=input_length) == 0)
                mask_5 = mask_temp_one_hot[:, 0, :].int().float()
                if (mask_temp_one_hot.size(1) >= 2):
                    for i in range(1, mask_temp_one_hot.size(1)):
                        mask_5 = mask_5 * (mask_temp_one_hot[:, i, :].int().float())
                mask_5 = mask_5.to(device)
                data_5 = data * mask_5

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

                torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.5)
                torch.nn.utils.clip_grad_norm_(header.parameters(), 1.5)
                torch.nn.utils.clip_grad_norm_(classifier.parameters(), 1.5)

                optimizer.step()

                train_loss += loss.item()
                train_total += len(label)
                predict = torch.max(output, 1)[1]
                train_correct += (predict == label).sum()
                train_bar.set_description(
                    'Train Epoch: [{}/{}] Loss: {:.4f}  ACC:{:.4f}'.format(epoch, num_epoches, train_loss / train_total,
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

                # aucroc(output,label)

                test_bar.set_description(
                    'Test Epoch: [{}/{}] Loss: {:.4f} ACC:{:.4f} '.format(epoch, num_epoches,
                                                                          test_loss / test_total,
                                                                          test_correct / test_total))

        label_test = []
        label_predict_test = []
        score_predict_test = []
        test_bar = tqdm(test_loader)
        for data, label in test_bar:
            label = label != 0
            label = label.long().to(device)
            label_test.append(label.cpu().numpy())

            #data = data[:, random_index]
            data = data.float().to(device)
            output = encoder(data)
            output = classifier(output)
            predict = torch.max(output, 1)[1]
            label_predict_test.append(predict.cpu().numpy())
            score_predict_test.append(output.cpu().detach().numpy())

        label_test = np.concatenate(label_test)
        label_predict_test = np.concatenate(label_predict_test)
        score_predict_test = np.concatenate(score_predict_test)

        cm = confusion_matrix(label_test, label_predict_test)
        np.set_printoptions(precision=2)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        test_acc = accuracy_score(label_test, label_predict_test)
        test_precision = precision_score(label_test, label_predict_test, average='macro')
        test_recall = recall_score(label_test, label_predict_test, average='macro')
        test_f1 = f1_score(label_test, label_predict_test, average='macro')
        print("model name dnn nsl-kdd", file=result_file)
        print("测试集准确率为:" + str(test_acc), file=result_file)
        print("测试集精确率为:" + str(test_precision), file=result_file)
        print("测试集recall为:" + str(test_recall), file=result_file)
        print("测试集f1为:" + str(test_f1), file=result_file)
        encoder_path='./models/9_'+str(data_name)+"_"+str(model_name)+"_encoder.pkl"
        torch.save(encoder,encoder_path)

        header_path='./models/9_'+str(data_name)+"_"+str(model_name)+"_header.pkl"
        torch.save(header,header_path)

        classifier_path='./models/9_'+str(data_name)+"_"+str(model_name)+"_classifier.pkl"
        torch.save(classifier, classifier_path)

        result_file.close()



























