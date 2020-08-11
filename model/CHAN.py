import math
import torch
import torch.nn as nn
from .attention import Attention


class CHAN(nn.Module):
    def __init__(self,config):
        nn.Module.__init__(self)
        self.config=config

        self.conv1d_1=nn.Conv1d(self.config["in_channel"],self.config["conv1_channel"],kernel_size=5,stride=1,padding=2)
        self.max_pooling_1=nn.MaxPool1d(2,stride=2,padding=0)
        self.conv1d_2=nn.Conv1d(self.config["conv1_channel"],self.config["conv2_channel"],kernel_size=5,stride=1,padding=2)
        self.max_pooling_2=nn.MaxPool1d(2,stride=2,padding=0)
        self.self_attention=Attention(self.config["conv2_channel"],self.config["conv2_channel"],self.config["conv2_channel"])
        self.concept_attention=Attention(self.config["concept_dim"],self.config["conv2_channel"],self.config["conv2_channel"])
        self.transpose_conv1d_1=torch.nn.ConvTranspose1d(4*self.config["conv2_channel"],self.config["deconv1_channel"],kernel_size=4,stride=2,padding=1)
        self.transpose_conv1d_2=torch.nn.ConvTranspose1d(self.config["deconv1_channel"],self.config["deconv2_channel"],kernel_size=4,stride=2,padding=1)
        self.similarity_linear1=torch.nn.Linear(self.config["deconv2_channel"],self.config["similarity_dim"],bias=False)
        self.similarity_linear2=torch.nn.Linear(self.config["concept_dim"],self.config["similarity_dim"],bias=False)

        self.MLP=torch.nn.Linear(self.config["similarity_dim"],1)

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
        attention_mask=torch.zeros(batch_size,max_seg_num,int(max_seg_length/4),dtype=torch.bool).cuda()
        for i in range(batch_size):
            for j in range(len(seg_len[i])):
                for k in range(math.ceil(seg_len[i][j]/4.0)):
                    attention_mask[i][j][k]=1

        # (batch_size * max_seg_num) * max_seg_length/4 * max_seg_length/4
        attention_mask=attention_mask.view(batch_size*max_seg_num,-1).unsqueeze(1)

        # (batch_size * max_seg_num) * 1 * max_seg_length/4 * 256
        K=tmp2.unsqueeze(1)
        # (batch_size * max_seg_num) * max_seg_length/4 * 1 * 256
        Q=tmp2.unsqueeze(2)

        # (batch_size * max_seg_num) * max_seg_length/4 * max_seg_length/4
        self_attention_score=self.self_attention(Q,K,attention_mask)

        # (batch_size * max_seg_num) * max_seg_length/4 * max_seg_length/4 * 256
        self_attention_result=self_attention_score.unsqueeze(-1)*K

        # (batch_size * max_seg_num) * max_seg_length/4 * 256
        self_attention_result=self_attention_result.sum(-2)

        # (batch_size * max_seg_num) * max_seg_length/4 * max_seg_length/4
        concept1_attention_score=self.concept_attention(concept1.unsqueeze(1).unsqueeze(1).unsqueeze(1).expand(batch_size,max_seg_num,int(self.config["max_frame_num"]/4),1,self.config["concept_dim"]).contiguous().view(batch_size*max_seg_num,int(self.config["max_frame_num"]/4),1,self.config["concept_dim"]),K,attention_mask)
        concept2_attention_score=self.concept_attention(concept2.unsqueeze(1).unsqueeze(1).unsqueeze(1).expand(batch_size,max_seg_num,int(self.config["max_frame_num"]/4),1,self.config["concept_dim"]).contiguous().view(batch_size*max_seg_num,int(self.config["max_frame_num"]/4),1,self.config["concept_dim"]),K,attention_mask)

        # (batch_size * max_seg_num) * max_seg_length/4 * max_seg_length/4 * 256
        concept1_attention_result=concept1_attention_score.unsqueeze(-1)*K
        # (batch_size * max_seg_num) * max_seg_length/4 * 256
        concept1_attention_result=concept1_attention_result.sum(-2)

        # (batch_size * max_seg_num) * max_seg_length/4 * max_seg_length/4 * 256
        concept2_attention_result=concept2_attention_score.unsqueeze(-1)*K

        # (batch_size * max_seg_num) * max_seg_length/4 * 256
        concept2_attention_result=concept2_attention_result.sum(-2)

        # (batch_size * max_seg_num) * max_seg_length/4 * 1024
        attention_result=torch.cat((tmp2,self_attention_result,concept1_attention_result,concept2_attention_result),dim=-1)

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
        concept1_score=torch.sigmoid(concept1_score)
        concept2_score=torch.sigmoid(concept2_score)

        # batch_size * max_seg_num * max_seg_length
        concept1_score=concept1_score.squeeze(-1).view(batch_size,max_seg_num,max_seg_length)
        concept2_score=concept2_score.squeeze(-1).view(batch_size,max_seg_num,max_seg_length)

        # batch_size * max_seg_num * max_seg_length
        return concept1_score,concept2_score
