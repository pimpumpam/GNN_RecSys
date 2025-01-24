import torch
import torch.nn as nn

import dgl

class PredictionLayer(nn.Module):
    def __init__(self, cfg_model):
        super(PredictionLayer, self).__init__()

        self.cfg_model = cfg_model
        self.model = nn.ModuleList()
        
        for i, (module, args) in enumerate(cfg_model.predict_layer['architecture']):
            layer = eval(module)(*args)
            self.model.append(layer)

    def mlp_predictor(self, e):
        x = torch.cat([e.src['h'], e.dst['h']], 1)
        
        for layer in self.model:
            x = layer(x)
            
        return {'score': x}


    def forward(self, sub_graph, x):
        with sub_graph.local_scope():
            sub_graph.ndata['h'] = x
            
            for etype in sub_graph.canonical_etypes:
                if self.cfg_model.predict_layer['predict_method'] == 'dot':
                    sub_graph.apply_edges(dgl.function.u_dot_v('h', 'h', 'score'), etype=etype)
                    
                elif self.cfg_model.predict_layer['predict_method'] == 'linear':
                    sub_graph.apply_edges(self.mlp_predictor, etype=etype)
                    
            return sub_graph.edata['score']