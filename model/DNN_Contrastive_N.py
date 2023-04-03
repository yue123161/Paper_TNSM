from torch import nn
from torch.nn import functional as F

class Encoder(nn.Module):
    def __init__(self,p=0.0):
        super(Encoder, self).__init__()
        self.basic_model=nn.Sequential(
            nn.Linear(121, 256),
            nn.ReLU(),
            #nn.Dropout(p),
            nn.Linear(256, 128),
            nn.ReLU(),
            #nn.Dropout(p),
        )
        self.header=nn.Sequential(
            nn.Linear(128, 128,bias=False),
        )
    def forward(self,x):
        x=self.basic_model(x)
        x=self.header(x)
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
                nn.Linear(256, 128,bias=False)
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
    def __init__(self,num_classes=2):
        super(LinearClassifier, self).__init__()
        self.fc = nn.Linear(128, num_classes)
    def forward(self, feature):
        return self.fc(feature)