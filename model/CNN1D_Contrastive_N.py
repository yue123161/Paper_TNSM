import torch
from torch import nn, autograd
from torch.nn import functional as F

class Encoder(nn.Module):
    def __init__(self,p=0.0):
        super(Encoder, self).__init__()
        self.basic_model = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=121, padding=121 // 2),
            nn.ReLU(),
            #nn.Dropout(p),
            nn.MaxPool1d(kernel_size=5),
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=5, padding=5 // 2),
            nn.ReLU(),
            #nn.Dropout(p),
            nn.MaxPool1d(kernel_size=3),
        )

        self.header = nn.Sequential(
            nn.Linear(32 * 8, 256,bias=False),
        )
    def forward(self,x):
        x = x.view(-1, 1, x.size(-1))
        x = self.basic_model(x)
        x = x.view(-1, 32 * 8)
        x = self.header(x)
        return x

class Header(nn.Module):
    def __init__(self, header='linear'):
        super(Header, self).__init__()
        if header == 'linear':
            self.head = nn.Linear(256, 512, bias=False)
        elif header == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(256, 512),
                nn.ReLU(),
                nn.Linear(512, 256,bias=False)
            )
        else:
            raise NotImplementedError(
                'head not supported: {}'.format(header))

    def forward(self, x):
        #x = F.normalize(x, dim=-1)  # 对输入进行正则化
        feature = F.normalize(self.head(x), dim=1)
        return feature

#####
### 这里需要考虑需不需要对x进行normalize
#####
# 定义分类器网络
class LinearClassifier(nn.Module):
    def __init__(self,num_classes=2):
        super(LinearClassifier, self).__init__()
        self.fc = nn.Linear(256, num_classes)
    def forward(self, feature):
        return self.fc(feature)

if __name__ == '__main__':
    testTensor = autograd.Variable(torch.randint(0, 256, [128, 121]) + 0.0)

    print(testTensor.shape)

    encoder = Encoder()

    out = encoder(testTensor)

    print(out.shape)