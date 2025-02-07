import math
import numpy as np
import pandas as pd
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
    self.tf = nn.Linear(self.te_dim,self.out_dim)
    # self.tf_day = nn.Linear(7,self.out_dim//2)
    # self.tf_hour = nn.Linear(24,self.out_dim//2)
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
    # te_day = self.tf_day(te[:,:,:7])
    # te_hour = self.tf_hour(te[:,:,7:])
    # te = torch.cat((te_day,te_hour),dim=-1)
    te = seelf.tf(te)
    te = self.fc_t(te)
    spe = torch.cat((se, pe), dim=-1)
    spe = self.fc_spe(spe)
    ste = spe.unsqueeze(0).unsqueeze(0).expand(te.size(0), te.size(1), -1, -1) + te.unsqueeze(2).expand(-1, -1, spe.size(0), -1)  # (B,T,N,D)
    ste = self.fc_ste(ste)
    return ste



class STA(nn.Module):
    def __init__(self, heads, d_model, in_seq_len, n_feature, window_size, dropout=0.1):
        super(STA, self).__init__()
        self.d_model = d_model
        self.heads = heads
        self.n_feature = n_feature
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
        window_list = np.arange(-(window_size-1)//2,(window_size-1)//2+1,1)
        shift_list = []
        for i in window_list:
            tmp = idxs+i
            tmp[tmp<0] = tmp[tmp<0] + len(idxs)
            tmp[tmp>(self.in_seq_len-1)] = tmp[tmp>(self.in_seq_len-1)] - len(idxs)
            shift_list.append(tmp)
        shift_list = np.array(shift_list)
        return shift_list,window_list

    def forward(self, x, ste, adj):
        # print(x.shape)
        # x_ste = torch.cat((x, ste), dim=-1) # (B,T,N,2D)
        q = self.fc_q(x) # (B,T,N,D)
        k = self.fc_k(x) # (B,T,N,D)   
        v = self.fc_v(x) # (B,T,N,D)
        B = q.size(0)
        T = q.size(1)
        N = q.size(2)
        D = q.size(3)
        hdim = D//self.heads
        q = q.view(B, T, N, self.heads, hdim)
        k = k.view (B, T, N, self.heads, hdim)
        v = v.view(B, T, N, self.heads, hdim)
        q = q.permute(0,1,3,2,4)
        v = v.permute(0,1,3,2,4)
        k = k.permute(0,1,3,2,4) # (B,T,heads,N,hdim)
        h = []
        for ti in range(len(self.shift_list)):
            # print(self.shift_list[ti])
            q =  q/(hdim**0.5)
            k =  k[:,self.shift_list[ti],:,:,:]
            v =  v[:,self.shift_list[ti],:,:,:]
            a = torch.matmul(q,k.transpose(-1,-2)) # attention score
            a = F.softmax(a, dim=-1)
            # mask operation
            adj_item = adj.unsqueeze(2).expand(-1,-1,self.heads,-1,-1)
            adj_item = adj_item[:,self.shift_list[ti],:,:,:]
            a = a*adj_item # final attention score
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
    
class OUT(nn.Module):
    def __init__(self, num_nodes, in_seq_len, out_seq_len, d_model):
        super(OUT, self).__init__()
        self.num_nodes = num_nodes
        self.in_seq_len = in_seq_len
        self.out_seq_len = out_seq_len
        self.d_model = d_model
        self.fc = nn.Linear(self.d_model * 2, self.d_model)
        self.fc11 = nn.Linear(self.d_model, self.d_model)
        self.fc12 = nn.Linear(self.d_model, self.d_model)
        self.fc21 = nn.Linear(self.d_model, self.d_model)
        self.fc22 = nn.Linear(self.d_model, self.d_model)
        self.out = nn.Linear(self.d_model, 1)

    def _reshape(self, x):
        return x.view(x.size(0), self.in_seq_len, self.num_nodes, self.d_model) #(B,T,N,D)

    def _output(self, x):
        B,T,N,D = x.size()
        x_1 = x.unsqueeze(2).repeat(1,1,N,1,1)
        x_2 = x.unsqueeze(3).repeat(1,1,1,N,1)
        x_1 = self.fc11(x_1)
        x_1 = self.fc12(x_1)
        x_2 = self.fc21(x_2)
        x_2 = self.fc22(x_2)
        x_cat = torch.cat((x_2, x_1), dim=-1) 
        y = self.fc(x_cat)
        y = self.out(y)
        y = y.squeeze(-1)
        return y

    def forward(self, x):
        # output layer
        x = self._reshape(x)  # (B,T,N,D)
        x = self._output(x)  # (B,T,N,N)
        # x = F.relu(x)
        return x
    
    
class MSSTA(nn.Module):
    def __init__(
        self,
        num_nodes,
        n_feature,
        in_seq_len,
        out_seq_len,
        s_in_channels,
        p_in_channels,
        t_in_channels,
        hidden_channels,
        attention_window,
        d_model,
        heads,
        num_encoder_layers,
    ):
        super(MSSTA, self).__init__()
        self.num_nodes = num_nodes
        self.in_seq_len = in_seq_len
        self.out_seq_len = out_seq_len
        self.d_model = d_model
        self.heads = heads
        self.num_encoder_layers = num_encoder_layers
        self.n_feature = n_feature
        self.s_in_channels = s_in_channels
        self.p_in_channels = p_in_channels
        self.t_in_channels = t_in_channels
        self.hidden_channels = hidden_channels
        self.attention_window = attention_window
        self.fc_out = nn.Linear(self.d_model,self.num_nodes)
        self.fc_x = nn.Linear(self.num_nodes,self.d_model)
        self.fc = nn.Linear(self.d_model,self.num_nodes**2)
        self.ste = STE(self.s_in_channels,self.p_in_channels,self.t_in_channels,self.d_model)
        self.transformAttention = TransformAttention(self.heads, self.d_model)
        self.sta_encoder = nn.ModuleList([STA(self.heads,self.d_model,self.in_seq_len,self.n_feature,self.attention_window) for _ in range(self.num_encoder_layers)])
        self.sta_decoder = nn.ModuleList([STA(self.heads,self.d_model,self.in_seq_len,self.n_feature,self.attention_window) for _ in range(self.num_encoder_layers)])
        # self.out = OUT(self.num_nodes, self.in_seq_len, self.out_seq_len, self.d_model)
    def forward(self, x, te, se, pe, adj,z_adj):
        x = self.fc_x(x)
        ste = self.ste(se, pe, te[:,:te.size(1),:])
        ste_his = ste[:,:te.size(1)//2,:,:]
        ste_pred = ste[:,te.size(1)//2:,:,:]
        # encoder 
        # print(x.shape)
        for net in self.sta_encoder:
            x = net(x, ste_his, adj)
        x = self.transformAttention(x, ste_his, ste_pred)
        # decoder
        for net in self.sta_decoder:
            x = net(x, ste_pred,adj)
        # x = self.fc(x)
        x = self.fc_out(x)
        # x_trans = x.permute(0,1,3,2)
        # x = torch.matmul(x,x_trans)
        # x = self.out(x)
        # B,T,N,N = x.size()
        # x = x.reshape(B,T,N*N)
        # x = self.fc_out(x)
        # x = x.reshape(B,T,N,N)
        # x = x * z_adj
        return x
    
class STA_Full(nn.Module):
    def __init__(self, heads, d_model, in_seq_len, n_feature, window_size = 5, dropout=0.1):
        super(STA_Full, self).__init__()
        self.d_model = d_model
        self.heads = heads
        self.n_feature = n_feature
        self.hdim = self.d_model//self.heads
        self.window = window_size
        self.in_seq_len = in_seq_len
        self.dropout = dropout
        self.fc_q  = nn.Linear(2*self.d_model, self.d_model)
        self.fc_k  = nn.Linear(2*self.d_model, self.d_model)
        self.fc_v  = nn.Linear(2*self.d_model, self.d_model)
        self.dense = nn.Linear(self.d_model, self.d_model)
        self.LayerNorm = nn.LayerNorm(self.d_model)
        self.dropout = nn.Dropout(self.dropout)
        

    def forward(self, x, ste, adj):
        # print(x.shape)
        x_ste = torch.cat((x, ste), dim=-1) # (B,T,N,2D)
        q = self.fc_q(x_ste) # (B,T,N,D)
        k = self.fc_k(x_ste) # (B,T,N,D)   
        v = self.fc_v(x_ste) # (B,T,N,D)
        B = q.size(0)
        T = q.size(1)
        N = q.size(2)
        D = q.size(3)
        hdim = D//self.heads
        q = q.view(B, T, N, self.heads, hdim)
        k = k.view (B, T, N, self.heads, hdim)
        v = v.view(B, T, N, self.heads, hdim)
        q = q.permute(0,3,1,2,4)
        v = v.permute(0,3,1,2,4)
        k = k.permute(0,3,1,2,4) # (B,heads,T,N,hdim)
        q = q.reshape(B,self.heads,T*N,hdim)
        k = k.reshape(B,self.heads,T*N,hdim)
        v = v.reshape(B,self.heads,T*N,hdim)
        a = torch.matmul(q,k.transpose(-1,-2)) # attention score (B,heads,T*N,T*N)
        a = F.softmax(a, dim=-1)
        h = torch.matmul(a,v) # (B,heads,T*N,hdim)
        h = h.reshape(B,self.heads,T,N,hdim)
        h = h.permute(0,2,3,1,4)
        h = h.reshape(B,T,N,D)
        h = self.dense(h)
        h = self.dropout(h)
        h = self.LayerNorm(h + x)
        return h 
    
    
class STA_Full_Encoder(nn.Module):
    def __init__(self, heads, d_model, in_seq_len, n_feature, window_size = 7, dropout=0.1):
        super(STA_Full_Encoder, self).__init__()
        self.d_model = d_model
        self.heads = heads
        self.n_feature = n_feature
        self.hdim = self.d_model//self.heads
        self.window = window_size
        self.in_seq_len = in_seq_len
        self.dropout = dropout
        self.fc = nn.Linear(self.d_model, 1)
        self.fc1 = nn.Linear(1, self.d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(66, 6, batch_first=True)

    def forward(self, x, ste, adj):
        # x_ste = torch.cat((x, ste), dim=-1) # (B,T,N,2D)
        x = x.squeeze(-1) #(B,T,N)
        h = self.encoder_layer(x) #(B,T,N)
        h = h.unsqueeze(-1) #(B,T,N,1)
        h = self.fc1(h) #(B,T,N,D)
        return h 
    
    
class MSSTA_Full(nn.Module):
    def __init__(
        self,
        num_nodes,
        n_feature,
        in_seq_len,
        out_seq_len,
        s_in_channels,
        p_in_channels,
        t_in_channels,
        hidden_channels,
        d_model,
        heads,
        num_encoder_layers,
    ):
        super(MSSTA_Full, self).__init__()
        self.num_nodes = num_nodes
        self.in_seq_len = in_seq_len
        self.out_seq_len = out_seq_len
        self.d_model = d_model
        self.heads = heads
        self.num_encoder_layers = num_encoder_layers
        self.n_feature = n_feature
        self.s_in_channels = s_in_channels
        self.p_in_channels = p_in_channels
        self.t_in_channels = t_in_channels
        self.hidden_channels = hidden_channels
        self.fc = nn.Linear(self.d_model * 2, self.d_model)
        self.fc_x = nn.Linear(self.num_nodes,self.d_model)
        self.ste = STE(self.s_in_channels,self.p_in_channels,self.t_in_channels,self.d_model)
        self.transformAttention = TransformAttention(self.heads, self.d_model)
        self.sta_encoder = nn.ModuleList([STA_Full_Encoder(self.heads,self.d_model,self.in_seq_len,self.n_feature) for _ in range(self.num_encoder_layers)])
        self.sta_decoder = nn.ModuleList([STA_Full_Encoder(self.heads,self.d_model,self.in_seq_len,self.n_feature) for _ in range(self.num_encoder_layers)])
        self.out = OUT(self.num_nodes, self.in_seq_len, self.out_seq_len, self.d_model)
        
    def forward(self, x, te, se, pe, adj):
        x = self.fc_x(x)
        ste = self.ste(se, pe, te[:,:te.size(1),:])
        ste_his = ste[:,:te.size(1)//2,:,:]
        ste_pred = ste[:,te.size(1)//2:,:,:]
        # encoder 
        for net in self.sta_encoder:
            x = net(x, ste_his, adj)
        x = self.transformAttention(x, ste_his, ste_pred)
        # decoder
        for net in self.sta_decoder:
            x = net(x, ste_pred,adj)
        return x
    

    
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
    def __init__(self, heads, d_model, in_seq_len, num_nodes, num_encoder_layers):
        super(META, self).__init__()
        self.d_model = d_model
        self.heads = heads
        self.num_nodes = num_nodes
        self.in_seq_len = in_seq_len
        self.num_encoder_layers = num_encoder_layers
        self.ste = STE(128,13,31,self.d_model)
        self.transformAttention = TransformAttention(self.heads, self.d_model)
        self.dense = nn.Linear(self.num_nodes,self.d_model)
        self.fc = nn.Linear(self.d_model,self.num_nodes)
        self.encoder = nn.ModuleList([nn.TransformerEncoderLayer(self.d_model, self.heads, batch_first=True) for _ in range(self.num_encoder_layers)])
        # Positional encoding
        self.positional_encoding = PositionalEncoding(
            self.in_seq_len*self.num_nodes, self.d_model
        )

    def forward(self, x, te, se, pe, adj, z_adj):
        B,T,N,N = x.size()
        x = x.reshape(B,T*N,N) #(B,T,N*N)
        x = self.dense(x) # (B,T,D)
        # x = self.positional_encoding(x) #(B,T,D)
        ste = self.ste(se, pe, te[:,:te.size(1),:])
        ste_his = ste[:,:te.size(1)//2,:,:]
        ste_his = ste_his.reshape(B,T*N,-1)
        ste_pred = ste[:,te.size(1)//2:,:,:]
        ste_pred = ste_pred.reshape(B,T*N,-1)
        for net in self.encoder:
            x = x + ste_his
            x = net(x)
        x = x.reshape(B,T,N,-1)
        ste_his = ste_his.reshape(B,T,N,-1)
        ste_pred = ste_pred.reshape(B,T,N,-1)
        x = self.transformAttention(x, ste_his, ste_pred)
        x = x.reshape(B,T*N,-1)
        ste_his = ste_his.reshape(B,T*N,-1)
        ste_pred = ste_pred.reshape(B,T*N,-1)
        for net in self.encoder:
            x = x + ste_pred
            h = net(x)
        h1 = h.permute(0,1,3,2)
        h = h*h1
        # h = self.fc(h)
        # h = h.reshape(B,T,N,N)
        h = h*z_adj
        return h
    
    
class META_CD(nn.Module):
    def __init__(self, heads, d_model, in_seq_len, num_nodes, num_encoder_layers):
        super(META_CD, self).__init__()
        self.d_model = d_model
        self.heads = heads
        self.num_nodes = num_nodes
        self.in_seq_len = in_seq_len
        self.num_encoder_layers = num_encoder_layers
        self.dense = nn.Linear(self.num_nodes**2,self.d_model)
        self.dense_c = nn.Linear(self.num_nodes**2,self.d_model)
        self.fc = nn.Linear(self.d_model,self.num_nodes**2)
        self.full_encoder_layer = nn.TransformerEncoderLayer(self.d_model, self.heads, batch_first=True)
        self.cd_encoder_layer = nn.TransformerEncoderLayer(self.d_model, self.heads, batch_first=True)
        # self.
        # Positional encoding
        self.positional_encoding = PositionalEncoding(
            self.in_seq_len, self.d_model
        )
        self.positional_encoding_c = PositionalEncoding(
            self.in_seq_len//2, self.d_model
        )

    def forward(self, x):
        B,T,N,N = x.size()
        x_c = x[:,T//2:,:,:]
        x = x.reshape(B,T,N*N) #(B,T,N*N)
        x_c = x_c.reshape(B,T//2,N*N)
        x = self.dense(x) # (B,T,D)
        x_c = self.dense_c(x_c)
        x = self.positional_encoding(x) #(B,T,D)
        x_c = self.positional_encoding(x_c)
        for i in range(self.num_encoder_layers):
            h = self.full_encoder_layer(x)
            h_c = self.cd_encoder_layer(x_c)
            h = torch.cat([h[:,:T//2,:],h[:,T//2:,:]+h_c],dim=1)
        h = self.fc(h)
        h = h.reshape(B,T,N,N)
        return h
    
class META_CDW(nn.Module):
    def __init__(self, heads, d_model, in_seq_len, num_nodes, num_encoder_layers, his_window):
        super(META_CDW, self).__init__()
        self.d_model = d_model
        self.heads = heads
        self.num_nodes = num_nodes
        self.in_seq_len = in_seq_len
        self.num_encoder_layers = num_encoder_layers
        self.his_window = his_window
        self.dense = nn.Linear(self.num_nodes**2,self.d_model)
        self.dense_c = nn.Linear(self.num_nodes**2,self.d_model)
        self.fc = nn.Linear(self.d_model,self.num_nodes**2)
        self.full_encoder_layer = nn.TransformerEncoderLayer(self.d_model, self.heads, batch_first=True)
        self.cd_encoder_layer = nn.TransformerEncoderLayer(self.d_model, self.heads, batch_first=True)
        self.gru = nn.GRU(self.in_seq_len * self.d_model, self.in_seq_len * self.d_model, 1)
        # Positional encoding
        self.positional_encoding = PositionalEncoding(
            self.in_seq_len, self.d_model
        )
        self.positional_encoding_c = PositionalEncoding(
            self.in_seq_len//2, self.d_model
        )

    def forward(self, x, c):
        B,T,N,N = x.size()
        x = x.reshape(B,T,N*N) #(B,T,N*N)
        x = self.dense(x) # (B,T,D)
        c = c.permute(1,0,2,3,4) #(K,B,T,N,N)
        x = self.positional_encoding(x) #(B,T,D)
        for i in range(self.num_encoder_layers):
            h = self.full_encoder_layer(x)
            h_w = []
            for j in range(self.his_window):
                r = c[j].reshape(B,T,N*N)
                r = self.dense_c(r) # (B,T,D)
                r = self.positional_encoding(r) # (B,T,D)
                r = self.cd_encoder_layer(r) 
                h_w.append(r)
            h_w = torch.stack(h_w,dim=0)# (K,B,T,D)
            h_w = h_w.reshape(self.his_window,B,T*self.d_model) # (K,B,T*D)
            h_w, _ = self.gru(h_w)
            h_w = h_w[-1]
            h_w = h_w.reshape(B,T,self.d_model)
            h = h + h_w    
        h = self.fc(h)
        h = h.reshape(B,T,N,N)
        return h
    
