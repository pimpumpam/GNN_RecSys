import torch
import torch.nn as nn


class ProjectionLayer(nn.Module):
    def __init__(self, cfg_model):
        super(ProjectionLayer, self).__init__()
        self.model = nn.ModuleDict()
        
        for key, val in cfg_model.projection_layer['architecture'].items():
            layers = nn.Sequential()
            
            for i, (module, args) in enumerate(val):
                layer = eval(module)(*args)
                layers.add_module(f"Layer #{i+1}", layer)
                
            self.model[f'{key}_no'] = layers

    def forward(self, blocks):
        x = blocks[0].srcdata['feature']
        
        for key, layer in self.model.items():
            
            x[key] = layer(x[key])
            
        blocks[0].srcdata['feature'] = x
        
        return blocks
