import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

class OUT(nn.Module):
    def __init__(self, num_nodes, in_seq_len, out_seq_len, d_model):
        super(OUT, self).__init__()
        self.num_nodes = num_nodes
        self.in_seq_len = in_seq_len
        self.out_seq_len = out_seq_len
        self.d_model = d_model
        self.fc = nn.Linear(self.d_model * 2, self.d_model)
        self.out = nn.Linear(self.d_model, 1)

    def _reshape(self, x):
        return x.view(x.size(0), self.in_seq_len, self.num_nodes, self.d_model) #(B,T,N,D)

    def _output(self, x):
        x_flat = x.unsqueeze(3)  # (B,T,N,_, D)
        x_repeat = x_flat.repeat(1, 1, 1, x.size(2), 1) #(B,T,N,N,D)
        x_reshape = torch.cat([x_repeat, x_repeat.transpose(2, 3)], dim=4)
        y = self.fc(x_reshape)
        y = self.out(y)
        y = y.squeeze(-1)
        return y

    def forward(self, x):
        # output layer
        x = self._reshape(x)  # (B,T,N,D)
        x = self._output(x)  # (B,T,N,N)
        # x = F.relu(x)
        return x
    
    
class STE(nn.Module):
  
  def __init__(self,se_dim,pe_dim,te_dim,out_dim):
    super(STE, self).__init__()
    self.se_dim = se_dim
    self.pe_dim = pe_dim
    self.te_dim = te_dim
    self.out_dim = out_dim
    self.sf = nn.Linear(self.se_dim,self.out_dim)
    self.pf = nn.Linear(self.pe_dim,self.out_dim)
    self.tf = nn.Linear(self.te_dim,self.out_dim)
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
    te = self.tf(te)
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
        self.mlp_q = nn.Linear(1,self.d_model)
        self.mlp_k = nn.Linear(1,self.d_model)
        self.mlp_v = nn.Linear(1,self.d_model)
        self.out = nn.Linear(self.d_model,1)

    def forward(self, x, ste_his, ste_pred):
        x = x.unsqueeze(-1) # (B,T,N*N,1)
        ste_his = ste_his.unsqueeze(-1)# (B,T,N*N,1)
        ste_pred = ste_pred.unsqueeze(-1)# (B,T,N*N,1)
        B = x.size(0)
        q = self.mlp_q(ste_pred)# (B,T,N*N,D)
        k = self.mlp_k(ste_his)# (B,T,N*N,D)
        v = self.mlp_v(x)# (B,T,N*N,D)
        q = torch.cat(torch.split(q, 8, dim=-1), dim=0)
        k = torch.cat(torch.split(k, 8, dim=-1), dim=0)
        v = torch.cat(torch.split(v, 8, dim=-1), dim=0)
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
        out = out.squeeze(-1)
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


class META(nn.Module):
    def __init__(self, heads, in_seq_len, num_edge, d_model, s_in_channels, p_in_channels, t_in_channels, num_encoder_layers):
        super(META, self).__init__()
        self.heads = heads
        self.num_edge = num_edge
        self.in_seq_len = in_seq_len
        self.d_model = d_model
        self.num_encoder_layers = num_encoder_layers
        self.s_in_channels = s_in_channels
        self.p_in_channels = p_in_channels
        self.t_in_channels = t_in_channels
        
        self.encoder = nn.ModuleList([nn.TransformerEncoderLayer(self.num_edge, self.heads, batch_first=True) for _ in range(self.num_encoder_layers)])
        self.fc = nn.Linear(self.num_edge, self.d_model)
        self.fc1 = nn.Linear(self.d_model, self.num_edge)
        # Positional encoding
        self.positional_encoding = PositionalEncoding(
            self.in_seq_len, self.d_model
        )
        self.ste = STE(self.s_in_channels,self.p_in_channels,self.t_in_channels,self.d_model)
        self.transformAttention = TransformAttention(self.heads, self.d_model)
        
    def _edge_ste(self,ste):
        ste = ste.unsqueeze(3).repeat(1, 1, 1, ste.size(2), 1) #(B,T,N,N,D)
        ste = torch.cat([ste, ste.transpose(2, 3)], dim=4) #(B,T,N,N,2D)
        ste = self.fc(ste) #(B,T,N,N,1)
        ste = ste.squeeze(-1) #(B,T,N,N)
        return ste

    def forward(self, x, te, se, pe, adj):
        x = x[:,:,:,:66] # (B,T,N,N)
        # ste = self.ste(se, pe, te[:,:te.size(1),:])
        # ste_his = ste[:,:te.size(1)//2,:,:]
        # ste_pred = ste[:,te.size(1)//2:,:,:]
        # ste_his = self._edge_ste(ste_his)
        # ste_pred = self._edge_ste(ste_pred)
        B,T,N,N = x.size()
        x = x.reshape(B,T,N*N) #(B,T,N*N)
        # x = self.fc(x) # (B,T,D)
        # ste_his = ste_his.reshape(B,T,N*N)
        # ste_pred = ste_pred.reshape(B,T,N*N)
        # x = self.positional_encoding(x) #(B,T,N*N)
        for net in self.encoder:
            # x = x + ste_his
            x = net(x)
        # x = self.transformAttention(x, ste_his, ste_pred)
        # for net in self.encoder:
        #     x = x + ste_pred
        #     x = net(x)
        # x = self.fc1(x)
        h = x.reshape(B,T,N,N)
        return h 
    

class MVSTA(nn.Module):
    def __init__(self, heads, in_seq_len, num_node, n_feature, d_model, s_in_channels, p_in_channels, t_in_channels, num_encoder_layers):
        super(MVSTA, self).__init__()
        self.heads = heads
        self.num_node = num_node
        self.in_seq_len = in_seq_len
        self.d_model = d_model
        self.s_in_channels = s_in_channels
        self.p_in_channels = p_in_channels
        self.t_in_channels = t_in_channels
        self.n_feature = n_feature
        self.num_encoder_layers = num_encoder_layers
        self.out = OUT(self.num_node, self.in_seq_len, self.in_seq_len, self.d_model)
        self.encoder = nn.ModuleList([nn.TransformerEncoderLayer(self.d_model, self.heads, batch_first=True) for _ in range(self.num_encoder_layers)])
        self.fc = nn.Linear(self.n_feature, self.d_model)
        self.fc_i = nn.Linear(self.num_node*2,self.num_node)
        # Positional encoding
        self.positional_encoding = PositionalEncoding(
            self.in_seq_len, self.d_model
        )

    def forward(self, x, te, se, pe, adj):
        x = self.fc(x) # (B,T,N,D)
        B,T,N,D = x.size()
        h =[]
        for i in range(T):
            if i == 0:
                x_i = torch.cat((x[:,i:i+1,:,:],x[:,i:i+1,:,:]),dim=1)
            else:
                x_i = x[:,i-1:i+1,:,:]
            x_i = x_i.reshape(B,2*N,D)
            for net in self.encoder:
                h_i = net(x_i) # (B,2*N,D)
                h_i = h_i.permute(0,2,1) #(B,D,2*N)
                h_i = self.fc_i(h_i) # (B, D, N)
                h_i = h_i.permute(0,2,1)
            h.append(h_i)
        h = torch.stack(h,dim=1) # (B,T,N,D)
        h = self.out(h)
        h = F.relu(h)
        return h 