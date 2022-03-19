import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv

class GCN(nn.Module):
    def __init__(self, in_feats, n_classes):
        super(GCN, self).__init__()
        # input layer
        self.layer1 = GraphConv(in_feats, 128, activation=F.relu)
        # hidden layers
        self.layer2 = GraphConv(128, 32, activation=F.relu)
        self.layer2_2 = GraphConv(32, 16, activation=F.relu)
        # output layer
        self.layer3 = GraphConv(16, n_classes)
        self.dropout = nn.Dropout(p=0.5)
    
    def forward(self, g, x):
        h = self.layer1(g, x)
        h = self.dropout(h)
        h = self.layer2(g, h)
        h = self.layer2_2(g, h)
        h = self.layer3(g, h)
        return h
