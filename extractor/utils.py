import torch
import torchvision
import torch.nn as nn
from torchvision import transforms


class Classifier(nn.Module):
    """ Naive implementation of classifier """
    def __init__(self, feat_in, n_classes):
        super(Classifier, self).__init__()
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(feat_in, n_classes)

    def forward(self, x):
        # x = self.avgpool(x)
        # x = torch.flatten(x, 1)        
        x = self.fc(x)
        return x

class FFN(nn.Module):
    """  """
    def __init__(self, feat_in, n_class, expansion=4, drop_p=0.1):
        super(FFN, self).__init__()
        self.fc1 = nn.Linear(feat_in, feat_in * expansion)
        self.fc2 = nn.Linear(feat_in * expansion, n_class)
        self.act = nn.ReLU()  # nn.GELU()  nn.SiLU()
        self.dropout_1 = nn.Dropout(drop_p)
        self.dropout_2 = nn.Dropout(drop_p)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout_1(x)
        x = self.fc2(x)
        return self.dropout_2(x)



'''
class kernel_node_generator(node_generator):
    """ Generate node for Kernel-BLS """

    def __init__(self):
        node_generator.__init__(self)
        self.normalscaler = MinMaxScaler()

    def feature_transform(self, testdata):
        '''  '''
        testnodes = self.nonlinear(testdata.dot(self.Wlist[0]) + self.blist[0])
        for i in range(1, len(self.Wlist)):
            testnodes = np.column_stack((testnodes, self.nonlinear(testdata.dot(self.Wlist[i]) + self.blist[i])))
        return testnodes

    def enhance_transform(self, testdata):
        '''  '''
        testnodes = self.nonlinear(testdata.dot(self.Wlist[0]) + self.blist[0])
        for i in range(1, len(self.Wlist)):
            testnodes = np.column_stack((testnodes, self.nonlinear(testdata.dot(self.Wlist[i]) + self.blist[i])))
        return testnodes
'''

def valid_model(model, X, y):
    ''' valid the model on dataset '''
    correct, total = 0, 0
    predict = model.predict(X)
    total += y.size
    correct += (predict == y).sum().item()
    accuracy = 100 * correct / total
    return accuracy


def extract_data(data_loader):
    ''' extract data from pytorch loader and return numpy list '''
    X, Y = [], []  # 
    for x, y in data_loader:
        X = x.numpy()
        Y = y.numpy()
    return X, Y 
