import dgl
import torch.nn as nn
from dgl.nn import GraphConv


class NodeClassificationHead(nn.Module):
    def __init__(self, in_feats, num_classes):
        super(NodeClassificationHead, self).__init__()
        self.conv = GraphConv(in_feats, num_classes)

    def forward(self, g, in_feat):
        h = self.conv(g, in_feat)
        return h


class GraphClassificationHead(nn.Module):
    def __init__(self, in_feats, num_classes):
        super(GraphClassificationHead, self).__init__()
        self.conv = GraphConv(in_feats, num_classes)

    def forward(self, g, in_feat):
        h = self.conv(g, in_feat)
        g.ndata["h"] = h
        h = dgl.mean_nodes(g, "h")
        return h
