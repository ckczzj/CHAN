# coding: utf-8

import math
import torch as t
from torch import nn
import torch.nn.functional as F
from config.config import DefaultConfig as config

INFINITY=1e15

class Attention(nn.Module):
    def __init__(self,Q_in_features_dim,K_in_features_dim,attention_dim):
        nn.Module.__init__(self)
        self.linear1=nn.Linear(Q_in_features_dim,attention_dim,bias=False)
        self.linear2=nn.Linear(K_in_features_dim,attention_dim,bias=False)
        self.linear3=nn.Linear(attention_dim,1,bias=False)

    def forward(self,Q,K,mask):
        attention_score=self.linear3(t.tanh(self.linear1(Q)+self.linear2(K))).squeeze(-1)
        attention_score=attention_score.masked_fill(mask==0,-INFINITY)
        attention_score=F.softmax(attention_score,dim=-1)
        return attention_score

class CHAN_without_SA(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)

        self.conv1d_1=nn.Conv1d(config.IN_CHANNEL,config.conv1_channel,kernel_size=5,stride=1,padding=2)
        self.max_pooling_1=nn.MaxPool1d(2,stride=2,padding=0)
        self.conv1d_2=nn.Conv1d(config.conv1_channel,config.conv2_channel,kernel_size=5,stride=1,padding=2)
        self.max_pooling_2=nn.MaxPool1d(2,stride=2,padding=0)
        self.self_attention=Attention(config.conv2_channel,config.conv2_channel,config.conv2_channel)
        self.concept_attention=Attention(config.CONCEPT_DIM,config.conv2_channel,config.conv2_channel)
        self.transpose_conv1d_1=t.nn.ConvTranspose1d(3*config.conv2_channel,config.deconv1_channel,kernel_size=4,stride=2,padding=1)
        self.transpose_conv1d_2=t.nn.ConvTranspose1d(config.deconv1_channel,config.deconv2_channel,kernel_size=4,stride=2,padding=1)
        self.similarity_linear1=t.nn.Linear(config.deconv2_channel,config.SIMILARITY_DIM,bias=False)
        self.similarity_linear2=t.nn.Linear(config.CONCEPT_DIM,config.SIMILARITY_DIM,bias=False)

        self.MLP=t.nn.Linear(config.SIMILARITY_DIM,1)

    # batch tensor: batch_size * max_seg_num * max_seg_length * 2048/4096
    # seg_len list(list(int)) : batch_size * seg_num (num of frame)
    # concept : batch_size * 300
    def forward(self,batch,seg_len,concept1,concept2):
        batch_size=batch.size()[0]
        max_seg_num=batch.size()[1]
        max_seg_length=batch.size()[2]

        # (batch_size * max_seg_num) * 128 * max_seg_length
        tmp1=self.conv1d_1(batch.view(batch_size*max_seg_num,max_seg_length,-1).transpose(1,2))
        # (batch_size * max_seg_num) * 128 * max_seg_length/2
        tmp1=self.max_pooling_1(tmp1)

        # (batch_size * max_seg_num) * 256 * max_seg_length/2
        tmp2=self.conv1d_2(tmp1)

        # (batch_size * max_seg_num) * max_seg_length/4 * 256
        tmp2=self.max_pooling_2(tmp2).transpose(1,2)

        # batch_size * max_seg_num * max_seg_length/4
        attention_mask=t.zeros(batch_size,max_seg_num,int(max_seg_length/4),dtype=t.uint8).cuda()
        for i in range(batch_size):
            for j in range(len(seg_len[i])):
                for k in range(math.ceil(seg_len[i][j]/4)):
                    attention_mask[i][j][k]=1

        # (batch_size * max_seg_num) * max_seg_length/4 * max_seg_length/4
        attention_mask=attention_mask.view(batch_size*max_seg_num,-1).unsqueeze(1)

        # (batch_size * max_seg_num) * 1 * max_seg_length/4 * 256
        K=tmp2.unsqueeze(1)

        # (batch_size * max_seg_num) * max_seg_length/4 * max_seg_length/4
        concept1_attention_score=self.concept_attention(concept1.unsqueeze(1).unsqueeze(1).unsqueeze(1).expand(batch_size,max_seg_num,int(config.MAX_FRAME_NUM/4),1,config.CONCEPT_DIM).contiguous().view(batch_size*max_seg_num,int(config.MAX_FRAME_NUM/4),1,config.CONCEPT_DIM),K,attention_mask)
        concept2_attention_score=self.concept_attention(concept2.unsqueeze(1).unsqueeze(1).unsqueeze(1).expand(batch_size,max_seg_num,int(config.MAX_FRAME_NUM/4),1,config.CONCEPT_DIM).contiguous().view(batch_size*max_seg_num,int(config.MAX_FRAME_NUM/4),1,config.CONCEPT_DIM),K,attention_mask)

        # (batch_size * max_seg_num) * max_seg_length/4 * max_seg_length/4 * 256
        concept1_attention_result=concept1_attention_score.unsqueeze(-1)*K
        # (batch_size * max_seg_num) * max_seg_length/4 * 256
        concept1_attention_result=concept1_attention_result.sum(-2)

        # (batch_size * max_seg_num) * max_seg_length/4 * max_seg_length/4 * 256
        concept2_attention_result=concept2_attention_score.unsqueeze(-1)*K

        # (batch_size * max_seg_num) * max_seg_length/4 * 256
        concept2_attention_result=concept2_attention_result.sum(-2)

        # (batch_size * max_seg_num) * max_seg_length/4 * 768
        attention_result=t.cat((tmp2,concept1_attention_result,concept2_attention_result),dim=-1)

        # (batch_size * max_seg_num) * 200 * max_seg_length/2
        result=self.transpose_conv1d_1(attention_result.transpose(1,2))

        # batch_size * max_seg_num * max_seg_length * 400
        result=self.transpose_conv1d_2(result).transpose(1,2).contiguous().view(batch_size,max_seg_num*max_seg_length,-1)

        # batch_size * (max_seg_num * max_seg_length) * 50
        similar_1=self.similarity_linear1(result)
        # batch_size * 50
        concept1_similar=self.similarity_linear2(concept1)
        concept2_similar=self.similarity_linear2(concept2)

        # batch_size * (max_seg_num * max_seg_length) * 50
        concept1_similar=concept1_similar.unsqueeze(1)*similar_1
        concept2_similar=concept2_similar.unsqueeze(1)*similar_1

        # batch_size * (max_seg_num * max_seg_length) * 1
        concept1_score=self.MLP(concept1_similar)
        concept2_score=self.MLP(concept2_similar)

        # batch_size * (max_seg_num * max_seg_length) * 1
        concept1_score=t.sigmoid(concept1_score)
        concept2_score=t.sigmoid(concept2_score)

        # batch_size * max_seg_num * max_seg_length
        concept1_score=concept1_score.squeeze(-1).view(batch_size,max_seg_num,max_seg_length)
        concept2_score=concept2_score.squeeze(-1).view(batch_size,max_seg_num,max_seg_length)

        # batch_size * max_seg_num * max_seg_length
        return concept1_score,concept2_score
