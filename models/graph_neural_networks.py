import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
import dgl.nn as dglnn

class GNNLayer(nn.Module):
    def __init__(self, cfg_model, etypes):
        super(GNNLayer, self).__init__()
        self.cfg_model = cfg_model
        self.model = nn.ModuleList()
        
        for module, args in cfg_model.gnn_layer['architecture']:
            for i, arg in enumerate(args):
                if isinstance(arg, str) and 'F.' in arg:
                    args[i] = eval(arg)
            
            module = eval(module)(*args)
            self.model.append(
                dglnn.HeteroGraphConv(
                    mods = {etype: module for etype in etypes},
                    aggregate = cfg_model.gnn_layer['aggregate_method']
                )
            )

    def forward(self, blocks, x):
        for i, (layer, block) in enumerate(zip(self.model, blocks)):
            x = layer(block, x)
            
            if self.cfg_model.name == 'Graph Attention Networks':
                for k, v in x.items():
                    if self.cfg_model.gnn_layer['multihead_merge_method'] == 'concat':
                        x[k] = v.view(v.size()[0], -1)
                        
                    else:
                        x[k] = torch.mean(v, dim=1)
        return x
