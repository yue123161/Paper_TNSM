from torch import nn
from torch.nn import functional as F

class Encoder(nn.Module):
    def __init__(self,p=0.0):
        super(Encoder, self).__init__()
        self.basic_model = nn.Sequential(
            nn.Conv2d(1, 64, (3, 3), padding=2),
            # nn.BatchNorm2d(64),
            nn.ReLU(),
            #nn.Dropout(p),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, (3, 3), padding=2),
            # nn.BatchNorm2d(128),
            nn.ReLU(),
            #nn.Dropout(p),
            nn.MaxPool2d(2, 2)
        )

        self.header = nn.Sequential(
            nn.Linear(128 * 16, 256,bias=False),
        )
    def forward(self,x):
        x = x.view(-1, 1, 11, 11)
        x = self.basic_model(x)
        x = x.view(-1, 128 * 16)
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