import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse import coo_matrix
from model_utils import STE, TransformAttention, PositionalEncoding, ContrastiveLoss


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj.float(), support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
    

class vsta_layer(nn.Module):
    def __init__(self, heads, d_model, in_seq_len, window_size, dropout=0.1):
        super(vsta_layer, self).__init__()
        self.d_model = d_model
        self.heads = heads
        self.hdim = self.d_model//self.heads
        self.window = window_size
        self.in_seq_len = in_seq_len
        self.dropout = dropout
        self.fc_q  = nn.Linear(self.d_model, self.d_model)
        self.fc_k  = nn.Linear(self.d_model, self.d_model)
        self.fc_v  = nn.Linear(self.d_model, self.d_model)
        self.dense = nn.Linear(self.in_seq_len*self.window, self.in_seq_len)
        self.LayerNorm = nn.LayerNorm(self.d_model)
        self.dropout = nn.Dropout(self.dropout)
        self.shift_list, self.window_list = self.get_shift_list()

    def get_shift_list(self):
        idxs = np.arange(self.in_seq_len) # [0,..,23]
        window_size = self.window 
        window_list = np.arange(-window_size,0,1)
        shift_list = []
        for i in window_list:
            tmp = idxs+i
            tmp[tmp<0] = tmp[tmp<0] + len(idxs)
            shift_list.append(tmp)
        shift_list = np.array(shift_list)
        return shift_list,window_list

    def forward(self, x, adj, s_adj):
        q = self.fc_q(x) # (B,T,N,D)
        k = self.fc_k(x) # (B,T,N,D)   
        v = self.fc_v(x) # (B,T,N,D)
        B,T,N,D = q.size()
        hdim = D//self.heads
        q = q.view(B, T, N, self.heads, hdim)
        k = k.view (B, T, N, self.heads, hdim)
        v = v.view(B, T, N, self.heads, hdim)
        q = q.permute(0,1,3,2,4)
        v = v.permute(0,1,3,2,4)
        k = k.permute(0,1,3,2,4) # (B,T,heads,N,hdim)
        h = []
        for ti in range(len(self.shift_list)):
            q =  q/(hdim**0.5)
            k =  k[:,self.shift_list[ti],:,:,:]
            v =  v[:,self.shift_list[ti],:,:,:]
            a = torch.matmul(q,k.transpose(-1,-2)) # attention score
            # mask operation
            adj_item = adj.unsqueeze(2).expand(-1,-1,self.heads,-1,-1) # (B,T,heads,N,N)
            adj_item = adj_item[:,self.shift_list[ti],:,:,:]
            # s_adj = s_adj.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(B, T, self.heads, N, N)
            # adj_item = torch.where(adj_item+s_adj>0,1,0)
            a = (a+1)*adj_item
            # a = torch.einsum('bthmn,mn->bthmn', a, s_adj) # final attention score
            a = F.softmax(a, dim=-1)
            r = torch.matmul(a,v)
            r = r.permute(0,1,3,2,4)
            r = r.reshape(B,T,N,D)
            h.append(r)
        h = torch.cat(h,dim=1)
        h = h.permute(0,2,3,1) #(B,N,D,3T)
        h = self.dense(h) #(B,N,D,T)
        h = h.permute(0,3,1,2)
        h = self.dropout(h)
        h = self.LayerNorm(h + x)
        return h
    

class vsa_layer(nn.Module):
    def __init__(self, heads, d_model, dropout=0.1):
        super(vsa_layer, self).__init__()
        self.heads = heads
        self.d_model = d_model
        self.hdim = self.d_model//self.heads
        self.dropout = dropout
        self.fc_q  = nn.Linear(self.d_model, self.d_model)
        self.fc_k  = nn.Linear(self.d_model, self.d_model)
        self.fc_v  = nn.Linear(self.d_model, self.d_model)
        self.LayerNorm = nn.LayerNorm(self.d_model)
        self.dropout = nn.Dropout(self.dropout)

    def forward(self, x, adj, s_adj):
        q = self.fc_q(x) # (B,T,N,D)
        k = self.fc_k(x) # (B,T,N,D)   
        v = self.fc_v(x) # (B,T,N,D)
        B,T,N,D = q.size()
        hdim = D//self.heads
        q = q.view(B, T, N, self.heads, hdim)
        k = k.view (B, T, N, self.heads, hdim)
        v = v.view(B, T, N, self.heads, hdim)
        q = q.permute(0,1,3,2,4)
        v = v.permute(0,1,3,2,4)
        k = k.permute(0,1,3,2,4) # (B,T,heads,N,hdim)
        a = torch.matmul(q,k.transpose(-1,-2))/(hdim**0.5)
        # mask operation
        adj_item = adj.unsqueeze(2).expand(-1,-1,self.heads,-1,-1) # (B,T,heads,N,N)
        # s_adj = s_adj.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(B, T, self.heads, N, N)
        # adj_item = torch.where(adj_item+s_adj>0,1,0)
        a = a*adj_item
        # a = torch.einsum('bthmn,mn->bthmn', a, s_adj) # final attention score
        a = F.softmax(a, dim=-1)
        h = torch.matmul(a,v)
        h = h.permute(0,1,3,2,4)
        h = h.reshape(B,T,N,D)
        h = self.dropout(h)
        h = self.LayerNorm(h + x)
        return h    
    
class ODTA_layer(nn.Module):
    def __init__(self, heads, d_model, in_seq_len, dropout=0.1):
        super(ODTA_layer, self).__init__()
        self.d_model = d_model
        self.heads = heads
        self.hdim = self.d_model//self.heads
        self.in_seq_len = in_seq_len
        self.dropout = dropout
        self.fc_q  = nn.Linear(self.d_model, self.d_model)
        self.fc_k  = nn.Linear(self.d_model, self.d_model)
        self.fc_v  = nn.Linear(self.d_model, self.d_model)
        self.LayerNorm = nn.LayerNorm(self.d_model)
        self.dropout = nn.Dropout(self.dropout)

    def forward(self, x):
        B,N,T,D = x.size()
        q = self.fc_q(x) # (B,N,T,D)
        k = self.fc_k(x) # (B,N,T,D)   
        v = self.fc_v(x) # (B,N,T,D)
        hdim = D//self.heads
        q = q.view(B, N, T, self.heads, hdim)
        k = k.view(B, N, T, self.heads, hdim)
        v = v.view(B, N, T, self.heads, hdim)
        q = q.permute(0,1,3,2,4)
        v = v.permute(0,1,3,2,4)
        k = k.permute(0,1,3,2,4) # (B,N,heads,T,hdim)
        a = torch.matmul(q,k.transpose(-1,-2))/(hdim**0.5)
        a = F.softmax(a, dim=-1)
        h = torch.matmul(a,v)
        h = h.permute(0,1,3,2,4) #(B,N,T,heads,hdim)
        h = h.reshape(B,N,T,D)
        h = self.dropout(h)
        h = self.LayerNorm(h + x)
        return h
    
class ODMixTA_layer(nn.Module):
    def __init__(self, heads, d_model, in_seq_len, dropout=0.1):
        super(ODMixTA_layer, self).__init__()
        self.d_model = d_model
        self.heads = heads
        self.hdim = self.d_model//self.heads
        self.in_seq_len = in_seq_len
        self.dropout = dropout
        self.fc_q  = nn.Linear(self.d_model, self.d_model)
        self.fc_k  = nn.Linear(self.d_model, self.d_model)
        self.fc_v  = nn.Linear(self.d_model, self.d_model)
        self.LayerNorm = nn.LayerNorm(self.d_model)
        self.dropout = nn.Dropout(self.dropout)

    def forward(self, x):
        B,T,D = x.size()
        q = self.fc_q(x) # (B,,T,D)
        k = self.fc_k(x) # (B,,T,D)   
        v = self.fc_v(x) # (B,,T,D)
        hdim = D//self.heads
        q = q.view(B, T, self.heads, hdim)
        k = k.view(B, T, self.heads, hdim)
        v = v.view(B, T, self.heads, hdim)
        q = q.permute(0,2,1,3)
        v = v.permute(0,2,1,3)
        k = k.permute(0,2,1,3) # (B,heads,T,hdim)
        a = torch.matmul(q,k.transpose(-1,-2))/(hdim**0.5)
        a = F.softmax(a, dim=-1)
        h = torch.matmul(a,v)
        h = h.permute(0,2,1,3) #(B,T,heads,hdim)
        h = h.reshape(B,T,D)
        h = self.dropout(h)
        h = self.LayerNorm(h + x)
        return h
    
class VSA_layer(nn.Module):
    def __init__(self, heads, d_model, in_seq_len, dropout=0.1):
        super(VSA_layer, self).__init__()
        self.d_model = d_model
        self.heads = heads
        self.hdim = self.d_model//self.heads
        self.in_seq_len = in_seq_len
        self.dropout = dropout
        self.fc_q  = nn.Linear(self.d_model, self.d_model)
        self.fc_k  = nn.Linear(self.d_model, self.d_model)
        self.fc_v  = nn.Linear(self.d_model, self.d_model)
        self.LayerNorm = nn.LayerNorm(self.d_model)
        self.dropout = nn.Dropout(self.dropout)

    def forward(self, x, adj):
        B,T,N,D = x.size()
        q = self.fc_q(x) # (B,T,N,D)
        k = self.fc_k(x) # (B,T,N,D)   
        v = self.fc_v(x) # (B,T,N,D)
        hdim = D//self.heads
        q = q.view(B, T, N, self.heads, hdim)
        k = k.view (B, T, N, self.heads, hdim)
        v = v.view(B, T, N, self.heads, hdim)
        q = q.permute(0,1,3,2,4)
        v = v.permute(0,1,3,2,4)
        k = k.permute(0,1,3,2,4) # (B,T,heads,N,hdim)
        a = torch.matmul(q,k.transpose(-1,-2))/(hdim**0.5)
        # mask operation
        adj = adj.unsqueeze(2).expand(-1,-1,self.heads,-1,-1) # (B,T,heads,N,N)
        a = a*adj # final attention score
        a = F.softmax(a, dim=-1)
        h = torch.matmul(a,v)
        h = h.permute(0,1,3,2,4) #(B,T,N,heads,hdim)
        h = h.reshape(B,T,N,D)
        h = self.dropout(h)
        h = self.LayerNorm(h + x)
        return h
    
    
class ESA_layer(nn.Module):
    def __init__(self, heads, d_model, in_seq_len, dropout=0.1):
        super(ESA_layer, self).__init__()
        self.d_model = d_model
        self.heads = heads
        self.hdim = self.d_model//self.heads
        self.in_seq_len = in_seq_len
        self.dropout = dropout
        self.fc_q  = nn.Linear(self.d_model, self.d_model)
        self.fc_k  = nn.Linear(self.d_model, self.d_model)
        self.fc_v  = nn.Linear(self.d_model, self.d_model)
        self.LayerNorm = nn.LayerNorm(self.d_model)
        self.dropout = nn.Dropout(self.dropout)

    def forward(self, x, adj):
        B,T,N,N,D = x.size()
        # x = x.reshape(B,T,N**2,D)
        # x_trans = x.permute(0,1,3,2,4)
        # x = torch.cat((x,x_trans),dim=3)
        q = self.fc_q(x) # (B,T,N,N,D)
        k = self.fc_k(x) # (B,T,N,N,D)
        v = self.fc_v(x) # (B,T,N,N,D)
        a = torch.matmul(q,k.transpose(-1,-2))/(D**0.5)
        # mask operation
        # adj = adj.unsqueeze(2).expand(-1,-1,self.heads,-1,-1) # (B,T,heads,N,N)
        # a = a*adj # final attention score
        a = F.softmax(a, dim=-1)
        h = torch.matmul(a,v)
        h = self.dropout(h)
        h = self.LayerNorm(h + x)
        return h