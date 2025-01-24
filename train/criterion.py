import torch
import torch.nn as nn


class HingeLoss(nn.Module):
    
    def __init__(self):
        super(HingeLoss, self).__init__()
        
    def forward(self, pos_score, neg_score, margin=0.5):
        """
        
        """
        pos_score = torch.cat(list(pos_score.values()))
        neg_score = torch.cat(list(neg_score.values()))
        
        n = pos_score.size()[0]
        
        return (neg_score.view(n, -1) - pos_score.view(n, -1) + margin).clamp(min=0).mean()
