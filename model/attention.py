import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self,Q_in_features_dim,K_in_features_dim,attention_dim):
        nn.Module.__init__(self)
        self.linear1=nn.Linear(Q_in_features_dim,attention_dim,bias=False)
        self.linear2=nn.Linear(K_in_features_dim,attention_dim,bias=False)
        self.linear3=nn.Linear(attention_dim,1,bias=False)

    def forward(self,Q,K,mask):
        attention_score=self.linear3(torch.tanh(self.linear1(Q)+self.linear2(K))).squeeze(-1)
        attention_score=attention_score.masked_fill(~mask,-1e10)
        attention_score=F.softmax(attention_score,dim=-1)
        return attention_score