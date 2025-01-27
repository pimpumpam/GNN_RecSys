import torch
import torch.nn as nn


class MaxMarginLoss(nn.Module):
    
    def __init__(self):
        super(MaxMarginLoss, self).__init__()
        
    def forward(self, pos_score, neg_score, margin=0.5):
        """
        Max margin loss 산출

        parameter
        ----------
        pos_score(torch.Tensor): Positive edge에 대한 모델 학습 score
        neg_score(torch.Tensor): Negative edge에 대한 모델 학습 score

        return
        ----------
        (float): 산출 된 max margin loss 값
        """
        pos_score = torch.cat(list(pos_score.values()))
        neg_score = torch.cat(list(neg_score.values()))
        
        n = pos_score.size()[0]
        
        return (neg_score.view(n, -1) - pos_score.view(n, -1) + margin).clamp(min=0).mean()
