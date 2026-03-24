import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GraphConv, SAGEConv, GATConv


class GCN(nn.Module):
    def __init__(self, in_feats, h_feats, depth):
        super(GCN, self).__init__()
        self.depth = depth
        self.conv1 = GraphConv(in_feats, h_feats)
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for i in range(depth-1):
            self.convs.append(GraphConv(h_feats, h_feats))
            self.bns.append(nn.BatchNorm1d(h_feats))

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        for i in range(self.depth-1):
            h = self.convs[i](g, h)
            h = F.relu(h)
            h = self.bns[i](h)
        return h


class GAT(nn.Module):
    def __init__(self, in_feats, h_feats, depth, heads=[2]):
        super().__init__()
        self.gat_layers = nn.ModuleList()
        self.depth = depth
        self.gat_layers.append(
            GATConv(
                in_feats,
                h_feats,
                heads[0],
                feat_drop=0.,
                attn_drop=0.,
                activation=F.elu,
            )
        )
        for i in range(depth):
            self.gat_layers.append(
                GATConv(
                    h_feats * heads[0],
                    h_feats,
                    heads[0],
                    feat_drop=0.5,
                    attn_drop=0.5,
                    activation=F.elu,
                )
            )
        
    def forward(self, g, inputs):
        h = inputs
        for i in range(self.depth):
            h = self.gat_layers[i](g, h)
            h = h.flatten(1)
        h = self.gat_layers[-1](g, h)
        h = h.mean(1)
        return h

class GraphSAGE(nn.Module):
    def __init__(self, in_feats, h_feats, depth):
        super(GraphSAGE, self).__init__()
        self.depth = depth
        self.conv1 = SAGEConv(in_feats, h_feats, 'mean')
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for i in range(depth-1):
            self.convs.append(SAGEConv(h_feats, h_feats, 'mean'))
            self.bns.append(nn.BatchNorm1d(h_feats))

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        for i in range(self.depth-1):
            if 'feat' in g.edata.keys():
                h = self.convs[i](g, h, edge_weight=g.edata['feat'])
            else:
                h = self.convs[i](g, h)
            h = F.relu(h)
            h = self.bns[i](h)
        return h
