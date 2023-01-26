import  torch
import  torch.nn as nn
import  dgl.nn as dglnn
from    dgl.nn.pytorch.conv import SAGEConv

class HomoModel(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super(HomoModel, self).__init__()
        self.conv1 = SAGEConv(in_feats, h_feats,     aggregator_type='mean', activation=torch.tanh)

        self.conv2 = SAGEConv(h_feats,  num_classes, aggregator_type='mean')

    def forward(self, mfgs, x):
        h_dst = x[:mfgs[0].num_dst_nodes()]
        h = self.conv1(mfgs[0], (x, h_dst))

        h_dst = h[:mfgs[1].num_dst_nodes()]
        h = self.conv2(mfgs[1], (h, h_dst))

        return h

class HeteroModel(nn.Module):
    def __init__(self, rel_shape_dict, hid_feats, out_feats):
        super().__init__()
        self.conv1 = dglnn.HeteroGraphConv({
            rel: dglnn.SAGEConv(size, hid_feats, aggregator_type='mean',activation=torch.tanh)
            for rel, size in rel_shape_dict.items()}, aggregate='mean')

        self.conv2 = dglnn.HeteroGraphConv({
            rel: dglnn.SAGEConv(hid_feats, out_feats,aggregator_type='mean',activation=torch.tanh)
            for rel in rel_shape_dict.keys()}, aggregate='mean')

    def forward(self, blocks):
        h = {k: v for k, v in blocks[0].srcdata['data'].items()}

        h = self.conv1(blocks[0], h)

        h = self.conv2(blocks[1], h)

        return h

## NOTE - the final models as implemented have had more complex implementations removed useful code framgments below.
# import  torch.nn.functional as F
# Feature normilsation function for SAGE CONV   -   norm = lambda x: F.normalize(x)
# Activation Function for Hetero Dictionary     -   x = {k: F.leaky_relu(v) for k, v in x.items()}
# Layer wise normilisation of hidden space      -   self.batch = nn.BatchNorm1d(h_feats) / h = self.batch(h)