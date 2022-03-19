import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphAttentionLayer, SpGraphAttentionLayer, Attention, Mlp, GraphConvolution, GraphAttentionLayer_2
from einops import rearrange, repeat

class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj[0]) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj[0]))
        return F.log_softmax(x, dim=1)

class GAT_3(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(GAT_3, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer_2(nhid, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.attentions2 = [GraphAttentionLayer_2(nhid, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions2):
            self.add_module('attention_{}'.format(i+nheads), attention)

        self.attentions3 = [GraphAttentionLayer_2(nhid * nheads, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions3):
            self.add_module('attention_{}'.format(i+nheads*2), attention)


        self.out_att = GraphAttentionLayer_2(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)
        self.gcn = GraphConvolution(nfeat, nhid)


    def forward(self, x, adj):
        # x = torch.mm(x, self.embed)
        # x = self.norm(x)

        x = F.dropout(x, self.dropout, training=self.training)
        # x = self.gcn(x, adj[0])
        # x = F.dropout(x, self.dropout/2, training=self.training)

        x = torch.cat([att(x, adj[1]) for ind, att in enumerate(self.attentions)], dim=1)
        # x2 = torch.cat([att(x, adj[1]-adj[0]) for ind, att in enumerate(self.attentions2)], dim=1)
        # x = x1 + x2 


        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj[0]))
        return F.log_softmax(x, dim=1)


class GAT_4(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(GAT_4, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer_2(nhid, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.attentions2 = [GraphAttentionLayer_2(nhid, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions2):
            self.add_module('attention_{}'.format(i+nheads), attention)

        self.attentions3 = [GraphAttentionLayer_2(nhid * nheads, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions3):
            self.add_module('attention_{}'.format(i+nheads*2), attention)
        # self.att1 = Attention(nhid * nheads)
        # self.mlp1 = Mlp(nhid * nheads)
        # self.att2 = Attention(nhid * nheads)
        # self.mlp2 = Mlp(nhid * nheads)

        self.out_att = GraphAttentionLayer_2(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)
        self.gcn = GraphConvolution(nfeat, nhid)

        # self.embed = nn.Parameter(torch.empty(size=(nfeat, nhid* nheads))) #//2)
        # nn.init.xavier_uniform_(self.embed.data, gain=1.414)
        # # self.out = nn.Parameter(torch.empty(size=(nhid* nheads, nhid))) #//2)
        # # nn.init.xavier_uniform_(self.out.data, gain=1.414)
        # # self.norm = nn.LayerNorm(nhid*nheads)
        # self.norm1 = nn.LayerNorm(nhid * nheads)
        # self.norm2 = nn.LayerNorm(nhid * nheads)
        # self.norm3 = nn.LayerNorm(nhid * nheads)
        # self.norm4 = nn.LayerNorm(nhid * nheads)
    def forward(self, x, adj):
        # x = torch.mm(x, self.embed)
        # x = self.norm(x)

        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gcn(x, adj[0])
        x = F.dropout(x, self.dropout/2, training=self.training)

        x1 = torch.cat([att(x, adj[1]) for ind, att in enumerate(self.attentions)], dim=1)
        x2 = torch.cat([att(x, adj[1]-adj[0]) for ind, att in enumerate(self.attentions2)], dim=1)
        x = x1 + x2 
        x = F.dropout(x, self.dropout/2, training=self.training)

        x = torch.cat([att(x, adj[0]) for ind, att in enumerate(self.attentions3)], dim=1) + x
        # x = rearrange(x, 'b d -> () b d')
        # x = self.att1(self.norm1(x), adj) + x
        # x = self.mlp1(self.norm2(x)) + x
        # x = self.att2(self.norm3(x), adj) + x
        # x = self.mlp2(self.norm4(x)) + x
        # x = rearrange(x, '() b d -> b d')


        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj[0]))
        return F.log_softmax(x, dim=1)


class GAT_2(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(GAT_2, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer_2(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.attentions2 = [GraphAttentionLayer_2(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions2):
            self.add_module('attention_{}'.format(i+nheads), attention)

        self.attentions3 = [GraphAttentionLayer_2(nhid * nheads, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions3):
            self.add_module('attention_{}'.format(i+nheads*2), attention)
        # self.att1 = Attention(nhid * nheads)
        # self.mlp1 = Mlp(nhid * nheads)
        # self.att2 = Attention(nhid * nheads)
        # self.mlp2 = Mlp(nhid * nheads)

        self.out_att = GraphAttentionLayer_2(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)
        # self.gcn = GraphConvolution(nfeat, nhid)

        # self.embed = nn.Parameter(torch.empty(size=(nfeat, nhid* nheads))) #//2)
        # nn.init.xavier_uniform_(self.embed.data, gain=1.414)
        # # self.out = nn.Parameter(torch.empty(size=(nhid* nheads, nhid))) #//2)
        # # nn.init.xavier_uniform_(self.out.data, gain=1.414)
        # # self.norm = nn.LayerNorm(nhid*nheads)
        # self.norm1 = nn.LayerNorm(nhid * nheads)
        # self.norm2 = nn.LayerNorm(nhid * nheads)
        # self.norm3 = nn.LayerNorm(nhid * nheads)
        # self.norm4 = nn.LayerNorm(nhid * nheads)
    def forward(self, x, adj):
        # x = torch.mm(x, self.embed)
        # x = self.norm(x)

        x = F.dropout(x, self.dropout, training=self.training)
        # x = self.gcn(x, adj[0])
        # x = F.dropout(x, self.dropout/2, training=self.training)

        x1 = torch.cat([att(x, adj[1]) for ind, att in enumerate(self.attentions)], dim=1)
        # x2 = torch.cat([att(x, adj[2]) for ind, att in enumerate(self.attentions2)], dim=1)
        x = x1 #+ x2 
        # x = F.dropout(x, self.dropout/2, training=self.training)

        # x = torch.cat([att(x, adj[0]) for ind, att in enumerate(self.attentions3)], dim=1) + x
        # x = rearrange(x, 'b d -> () b d')
        # x = self.att1(self.norm1(x), adj) + x
        # x = self.mlp1(self.norm2(x)) + x
        # x = self.att2(self.norm3(x), adj) + x
        # x = self.mlp2(self.norm4(x)) + x
        # x = rearrange(x, '() b d -> b d')


        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj[0]))
        return F.log_softmax(x, dim=1)



class SpGAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Sparse version of GAT."""
        super(SpGAT, self).__init__()
        self.dropout = dropout

        self.attentions = [SpGraphAttentionLayer(nfeat, 
                                                 nhid, 
                                                 dropout=dropout, 
                                                 alpha=alpha, 
                                                 concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = SpGraphAttentionLayer(nhid * nheads, 
                                             nclass, 
                                             dropout=dropout, 
                                             alpha=alpha, 
                                             concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return F.log_softmax(x, dim=1)

