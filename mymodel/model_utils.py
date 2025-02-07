import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class STE(nn.Module):
  
  def __init__(self,se_dim,pe_dim,te_dim,out_dim):
    super(STE, self).__init__()
    self.se_dim = se_dim
    self.pe_dim = pe_dim
    self.te_dim = te_dim
    self.out_dim = out_dim
    self.sf = nn.Linear(self.se_dim,self.out_dim)
    self.pf = nn.Linear(self.pe_dim,self.out_dim)
    # self.tf = nn.Linear(self.te_dim,self.out_dim)
    self.tf_day = nn.Linear(7,self.out_dim//2)
    self.tf_hour = nn.Linear(24,self.out_dim//2)
    self.fc_s = nn.Linear(self.out_dim,self.out_dim)
    self.fc_p = nn.Linear(self.out_dim,self.out_dim)
    self.fc_t = nn.Linear(self.out_dim,self.out_dim)
    self.fc_spe = nn.Linear(self.out_dim*2,self.out_dim)
    self.fc_ste = nn.Linear(self.out_dim,self.out_dim)
  def forward(self,se,pe,te):
    se = self.sf(se)
    se = self.fc_s(se)
    pe = self.pf(pe)
    pe = self.fc_p(pe)
    B,T,N = te.size()
    te_day = self.tf_day(te[:,:,:7])
    te_hour = self.tf_hour(te[:,:,7:])
    te_day = te_day.reshape(B,T,-1)
    te_hour = te_hour.reshape(B,T,-1)
    te = torch.cat((te_day,te_hour),dim=-1)
    # te = self.tf(te)
    te = self.fc_t(te)
    spe = torch.cat((se, pe), dim=-1)
    spe = self.fc_spe(spe)
    ste = spe.unsqueeze(0).unsqueeze(0).expand(te.size(0), te.size(1), -1, -1) + te.unsqueeze(2).expand(-1, -1, spe.size(0), -1)  # (B,T,N,D)
    ste = self.fc_ste(ste)
    return ste


class TransformAttention(nn.Module):
    def __init__(self, heads, d_model):
        super(TransformAttention, self).__init__()
        self.d_model = d_model
        self.heads = heads
        self.hdim = self.d_model//self.heads
        self.mlp_q = nn.Linear(self.d_model,self.d_model)
        self.mlp_k = nn.Linear(self.d_model,self.d_model)
        self.mlp_v = nn.Linear(self.d_model,self.d_model)
        self.out = nn.Linear(self.d_model,self.d_model)

    def forward(self, x, ste_his, ste_pred):
        B = x.size(0)
        q = self.mlp_q(ste_pred)
        k = self.mlp_k(ste_his)
        v = self.mlp_v(x)
        q = torch.cat(torch.split(q, self.heads, dim=-1), dim=0)
        k = torch.cat(torch.split(k, self.heads, dim=-1), dim=0)
        v = torch.cat(torch.split(v, self.heads, dim=-1), dim=0)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 3, 1)
        v = v.permute(0, 2, 1, 3)
        a = torch.matmul(q, k)
        a /= (self.hdim ** 0.5)
        a = F.softmax(a, dim=-1)
        out = torch.matmul(a, v)
        out = out.permute(0, 2, 1, 3)
        out = torch.cat(torch.split(out, B, dim=0), dim=-1)
        out = self.out(out)
        return out
    
    
class PositionalEncoding(nn.Module):
    def __init__(self, seq_length, d_model):
        super(PositionalEncoding, self).__init__()
        self.register_buffer(
            "positional_encoding",
            self._generate_positional_encoding(seq_length, d_model),
        )

    def forward(self, x):
        return x + self.positional_encoding[:, : x.size(1)]

    def _generate_positional_encoding(self, seq_length, d_model):
        pe = torch.zeros(seq_length, d_model)
        position = torch.arange(0, seq_length).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        return pe


class ContrastiveLoss(nn.Module):
    def __init__(self, margin):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, triplets):
        total_loss = 0
        for triplet in triplets:
            anchor, positive, negative = triplet
            distance_positive = torch.nn.functional.pairwise_distance(anchor, positive)
            distance_negative = torch.nn.functional.pairwise_distance(anchor, negative)
            loss = torch.mean(torch.clamp(self.margin + distance_positive - distance_negative, min=0.0))
            total_loss += loss
        total_loss = total_loss/len(triplets)
        return total_loss