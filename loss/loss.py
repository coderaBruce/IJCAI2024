import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class CosineContrastiveLoss(nn.Module):
    
    def __init__(self, config_dict):
        """CCL loss from SimpleX: https://arxiv.org/abs/2109.12613

        Args:
            config_dict: configuration dictionary
        """
        super().__init__()
        
        self._margin = config_dict['neg_margin']
        self._negative_weight = config_dict['neg_weight']

    def forward(self, pos_scores, neg_scores):
        
        pos_loss = torch.relu(1 - pos_scores)
        neg_loss = torch.relu(neg_scores - self._margin)
        if self._negative_weight:
            loss = pos_loss + neg_loss.mean(dim=-1) * self._negative_weight
        else:
            loss = pos_loss + neg_loss.sum(dim=-1)
            
        return loss.mean()

##  
class CosineContrastiveLoss_Debiased(nn.Module):

    def __init__(self, config_dict):
        super().__init__()
        
        self._margin = config_dict['neg_margin']
        self._negative_weight = config_dict['neg_weight']
        
            

    def forward(self, pos_scores, neg_scores, ext_scores, tu):
        
        pos_loss = tu * torch.relu(1 - pos_scores)
        neg_loss = torch.relu(neg_scores - self._margin).mean(-1)
        ext_loss = - tu * torch.relu(ext_scores - self._margin).mean(-1)
        debiased_neg_loss = neg_loss + ext_loss
        loss = pos_loss + self._negative_weight * debiased_neg_loss

        return loss.mean()


class Decoupling(nn.Module):

    def __init__(self, config_dict):
        
        super().__init__()
        
        
        self.epsilon = config_dict['pos_weight']

        self.t_pos = config_dict['pos_temperature']
        
        self.t_neg = config_dict['neg_temperature']
        

    def forward(self, pos_scores, neg_scores):
        

        pos_scores = (pos_scores)/self.t_pos


        neg_scores = (neg_scores)/self.t_neg


        neg = torch.log(torch.exp(neg_scores).sum(-1))
        
        
        loss = pos_scores - neg
        
        return - loss.mean()
    
