import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse import coo_matrix
from layers import vsta_layer, vsa_layer, ODTA_layer, ESA_layer, ODMixTA_layer, GraphConvolution
from model_utils import STE, TransformAttention, PositionalEncoding, ContrastiveLoss
from torch_geometric.nn import GATConv
from torch_geometric.data import Data


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nout, dropout):
        super(GCN, self).__init__()
        self.fc = nn.Linear(66,nfeat)
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nout)
        self.dropout = dropout

    def forward(self, x, adj):
        x = self.fc(x)
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return x
    
    
   
class VSTA(nn.Module):
    def __init__(
        self,
        num_nodes,
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
        super(VSTA, self).__init__()
        self.num_nodes = num_nodes # N
        self.in_seq_len = in_seq_len # T
        self.out_seq_len = out_seq_len # T
        self.d_model = d_model # D
        self.heads = heads 
        self.num_encoder_layers = num_encoder_layers 
        self.s_in_channels = s_in_channels
        self.p_in_channels = p_in_channels
        self.t_in_channels = t_in_channels
        self.hidden_channels = hidden_channels
        self.attention_window = attention_window
        self.fc_x = nn.Linear(self.num_nodes,self.d_model)
        self.fusion = nn.Linear(self.d_model*3,self.d_model)
        self.fusion_decoder = nn.Linear(self.d_model*2,self.d_model)
        self.x_out = nn.Linear(self.d_model,self.num_nodes)
        self.out = nn.Linear(self.d_model,self.num_nodes)
        self.ste = STE(self.s_in_channels,self.p_in_channels,self.t_in_channels,self.d_model)
        self.transformAttention = TransformAttention(self.heads, self.d_model)
        self.sta_encoder = nn.ModuleList([vsta_layer(self.heads,self.d_model,self.in_seq_len,self.attention_window) for _ in range(self.num_encoder_layers)])
        self.sta_decoder = nn.ModuleList([vsta_layer(self.heads,self.d_model,self.in_seq_len,self.attention_window) for _ in range(self.num_encoder_layers)])
    def forward(self, x, x_r, te, se, pe, adj,s_adj):
        # ste block
        ste = self.ste(se, pe, te[:,:te.size(1),:])
        ste_his = ste[:,:te.size(1)//2,:,:]
        ste_pred = ste[:,te.size(1)//2:,:,:]
        # ste fussion 
        x = self.fc_x(x)
        x = self.fusion(torch.cat((x,x_r,ste_his),dim=-1))
        # x = self.fusion(torch.cat((x,x_r),dim=-1))
        # encoder 
        for net in self.sta_encoder:
            x = net(x, adj, s_adj)
        # x = self.transformAttention(x, ste_his, ste_pred) # time alignment
        x_trans = x.transpose(-1,-2)
        x_adj = torch.matmul(x,x_trans)
        adj = torch.where(x_adj>0,1,0) # new generated adj
        x = self.fusion_decoder(torch.cat((x,ste_pred),dim=-1))
        # decoder
        for net in self.sta_decoder:
            x = net(x, adj, s_adj)
        x = self.out(x)
        # output
        # x_trans = x.transpose(-1,-2)
        # x = torch.matmul(x,x_trans)/(x.size(-1))
        return x
    
class VSA(nn.Module):
    def __init__(
        self,
        d_model,
        heads,
        num_encoder_layers,
    ):
        super(VSA, self).__init__()
        self.d_model = d_model # D
        self.heads = heads 
        self.num_encoder_layers = num_encoder_layers 
        self.fc = nn.Linear(66, d_model)
        self.out = nn.Linear(d_model, 66)
        self.vsa_layer = nn.ModuleList([vsa_layer(self.heads,self.d_model) for _ in range(self.num_encoder_layers)])
    def forward(self, x, adj, s_adj):
        x = self.fc(x)
        for net in self.vsa_layer:
            x = net(x, adj, s_adj)
        # x = self.out(x)
        return x
    
class ODTA(nn.Module):
    def __init__(
        self,
        num_nodes,
        in_seq_len,
        out_seq_len,
        d_model,
        heads,
        num_encoder_layers
    ):
        super(ODTA, self).__init__()
        self.num_nodes = num_nodes # N
        self.in_seq_len = in_seq_len # T
        self.out_seq_len = out_seq_len # T
        self.d_model = d_model # D
        self.heads = heads
        self.num_encoder_layers = num_encoder_layers 
        # self.fc = nn.Linear(self.,self.d_model)
        self.out = nn.Linear(self.d_model,self.num_nodes)
        self.positional_encoding = PositionalEncoding(
            self.in_seq_len, self.d_model
        )
        self.odta_layer = nn.ModuleList([ODTA_layer(self.heads,self.d_model,self.in_seq_len) for _ in range(self.num_encoder_layers)])
    def forward(self, x):
        # x = x.unsqueeze(-1) # (B,T,N,N,1)
        # x = self.fc(x) # (B,T,N,N,D)
        x = x.permute(0,2,1,3) # (B,N,T,D)
        x = self.positional_encoding(x)
        for net in self.odta_layer:
            x = net(x)    
        # output
        x = self.out(x)
        # x = x.squeeze(-1)
        x = x.permute(0,2,1,3)
        return x
    

class ODMixTA(nn.Module):
    def __init__(
        self,
        num_nodes,
        in_seq_len,
        out_seq_len,
        d_model,
        heads,
        num_encoder_layers
    ):
        super(ODMixTA, self).__init__()
        self.num_nodes = num_nodes # N
        self.in_seq_len = in_seq_len # T
        self.out_seq_len = out_seq_len # T
        self.d_model = d_model # D
        self.heads = heads
        self.num_encoder_layers = num_encoder_layers 
        self.fc = nn.Linear(self.num_nodes**2,self.d_model)
        self.out = nn.Linear(self.d_model,self.num_nodes**2)
        self.positional_encoding = PositionalEncoding(
            self.in_seq_len, self.d_model
        )
        self.odta = nn.ModuleList([ODMixTA_layer(self.heads,self.d_model,self.in_seq_len) for _ in range(self.num_encoder_layers)])
    def forward(self, x):
        B,T,N,N = x.size()
        x = x.reshape(B,T,N*N)
        x = self.fc(x) # (B,T,D)
        x = self.positional_encoding(x)
        for net in self.odta:
            x = net(x)    
        # output
        x = self.out(x)
        x = x.reshape(B,T,N,N)
        return x
    
    
class ESA(nn.Module):
    def __init__(
        self,
        heads,
        d_model,
        in_seq_len,
        num_encoder_layers
    ):
        super(ESA, self).__init__()
        self.in_seq_len = in_seq_len # T
        self.d_model = d_model # D
        self.heads = heads
        self.num_encoder_layers = num_encoder_layers 
        self.fc = nn.Linear(1,self.d_model)
        self.out = nn.Linear(self.d_model,1)
        self.esa = nn.ModuleList([ESA_layer(self.heads,self.d_model,self.in_seq_len) for _ in range(self.num_encoder_layers)])
        self.esa_d = nn.ModuleList([ESA_layer(self.heads,self.d_model,self.in_seq_len) for _ in range(self.num_encoder_layers)])
    def forward(self, x, te, se, pe, adj,s_adj):
        x = x.unsqueeze(-1) # (B,T,N,N,1)
        x = self.fc(x) # (B,T,N,N,D)
        x_d = x.transpose(-2,-3)
        for net in self.esa:
            x = net(x,adj) 
        for net in self.esa_d:
            x_d = net(x_d,adj)
        # output
        x = self.out(x)
        x_d = self.out(x_d)
        x = x + x_d
        x = x.squeeze(-1)
        return x
    

class BSTA(nn.Module):
    def __init__(
        self,
        num_nodes,
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
        super(BSTA, self).__init__()
        self.num_nodes = num_nodes # N
        self.in_seq_len = in_seq_len # T
        self.out_seq_len = out_seq_len # T
        self.d_model = d_model # D
        self.heads = heads 
        self.num_encoder_layers = num_encoder_layers 
        self.s_in_channels = s_in_channels
        self.p_in_channels = p_in_channels
        self.t_in_channels = t_in_channels
        self.hidden_channels = hidden_channels
        self.attention_window = attention_window
        self.in_sta = VSTA(self.num_nodes,self.in_seq_len,self.out_seq_len,self.s_in_channels,self.p_in_channels,self.t_in_channels,self.hidden_channels,self.attention_window,self.d_model,self.heads,self.num_encoder_layers)
        self.out_sta = VSTA(self.num_nodes,self.in_seq_len,self.out_seq_len,self.s_in_channels,self.p_in_channels,self.t_in_channels,self.hidden_channels,self.attention_window,self.d_model,self.heads,self.num_encoder_layers)
        # # self.odta = ODTA(self.num_nodes,self.in_seq_len,self.out_seq_len,self.d_model,self.heads,self.num_encoder_layers)
        # self.odlinear = ODLinear(self.num_nodes,self.in_seq_len, self.out_seq_len)
        
    def forward(self, x, x_r, te, se, pe, adj,s_adj):
        # inflow
        x_in = self.in_sta(x, x_r, te, se, pe, adj,s_adj)
        # outflow
        x_out = x.transpose(-1,-2)
        adj_out = adj.transpose(-1,-2)
        x_out = self.out_sta(x_out, x_r, te, se, pe, adj_out,s_adj)
        x_out = x_out.transpose(-1,-2)
        # fussion
        x = (x_in + x_out)/2
        # od
        # x_od = self.odta(x_v)
        # x_od = self.odlinear(x_v)
        return x
    
    
    
class WSTA(nn.Module):
    def __init__(
        self,
        num_nodes,
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
        super(WSTA, self).__init__()
        self.num_nodes = num_nodes # N
        self.in_seq_len = in_seq_len # T
        self.out_seq_len = out_seq_len # T
        self.d_model = d_model # D
        self.heads = heads 
        self.num_encoder_layers = num_encoder_layers 
        self.s_in_channels = s_in_channels
        self.p_in_channels = p_in_channels
        self.t_in_channels = t_in_channels
        self.hidden_channels = hidden_channels
        self.attention_window = attention_window
        self.bsta = BSTA(self.num_nodes,self.in_seq_len,self.out_seq_len,self.s_in_channels,self.p_in_channels,self.t_in_channels,self.hidden_channels,self.attention_window,self.d_model,self.heads,self.num_encoder_layers)
        
    def forward(self, x, te, se, pe, adj,c, tc, ac, s_adj):
        c = c.transpose(0,1)
        tc = tc.transpose(0,1)
        ac = ac.transpose(0,1)
        h = 0
        for w in range(c.size(0)):
            c_item = self.bsta(c[w], tc[w], se, pe, ac[w],s_adj)
            h += c_item
        h = h/c.size(0)
        return h
    
    
class ODLinear(nn.Module):
    """
    Just one Linear layer
    """
    def __init__(self, num_nodes, in_seq_len, out_seq_len, individual=True):
        super(ODLinear, self).__init__()
        self.in_seq_len = in_seq_len
        self.out_seq_len = out_seq_len
        
        # Use this line if you want to visualize the weights
        # self.Linear.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.out_seq_len,self.seq_len]))
        self.channels = num_nodes**2
        self.individual = individual
        if self.individual:
            self.Linear = nn.ModuleList()
            for i in range(self.channels):
                self.Linear.append(nn.Linear(self.in_seq_len,self.out_seq_len))
        else:
            self.Linear = nn.Linear(self.seq_len, self.out_seq_len)

    def forward(self, x, te, se, pe, adj,s_adj):
        B,T,N,N = x.size()
        x = x.reshape(B,T,N*N)
        if self.individual:
            output = torch.zeros([x.size(0),self.out_seq_len,x.size(2)],dtype=x.dtype).to(x.device)
            for i in range(self.channels):
                output[:,:,i] = self.Linear[i](x[:,:,i])
            x = output
        x = x.reshape(B,T,N,N)
        return x # [Batch, Output length, Channel]

class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

class ODDLinear(nn.Module):
    """
    Decomposition-Linear
    """
    def __init__(self, num_nodes, in_seq_len, out_seq_len, individual=True):
        super(ODDLinear, self).__init__()
        self.in_seq_len = in_seq_len
        self.out_seq_len = out_seq_len

        # Decompsition Kernel Size
        kernel_size = 25
        self.decompsition = series_decomp(kernel_size)
        self.individual = individual
        self.channels = num_nodes**2

        if self.individual:
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()
            
            for i in range(self.channels):
                self.Linear_Seasonal.append(nn.Linear(self.in_seq_len,self.out_seq_len))
                self.Linear_Trend.append(nn.Linear(self.in_seq_len,self.out_seq_len))

                # Use this two lines if you want to visualize the weights
                # self.Linear_Seasonal[i].weight = nn.Parameter((1/self.seq_len)*torch.ones([self.out_seq_len,self.seq_len]))
                # self.Linear_Trend[i].weight = nn.Parameter((1/self.seq_len)*torch.ones([self.out_seq_len,self.seq_len]))
        else:
            self.Linear_Seasonal = nn.Linear(self.in_seq_len,self.out_seq_len)
            self.Linear_Trend = nn.Linear(self.in_seq_len,self.out_seq_len)
            
            # Use this two lines if you want to visualize the weights
            # self.Linear_Seasonal.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.out_seq_len,self.seq_len]))
            # self.Linear_Trend.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.out_seq_len,self.seq_len]))

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        B,T,N,N = x.size()
        x = x.reshape(B,T,N*N)
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init, trend_init = seasonal_init.permute(0,2,1), trend_init.permute(0,2,1)
        if self.individual:
            seasonal_output = torch.zeros([seasonal_init.size(0),seasonal_init.size(1),self.out_seq_len],dtype=seasonal_init.dtype).to(seasonal_init.device)
            trend_output = torch.zeros([trend_init.size(0),trend_init.size(1),self.out_seq_len],dtype=trend_init.dtype).to(trend_init.device)
            for i in range(self.channels):
                seasonal_output[:,i,:] = self.Linear_Seasonal[i](seasonal_init[:,i,:])
                trend_output[:,i,:] = self.Linear_Trend[i](trend_init[:,i,:])
        else:
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)

        x = seasonal_output + trend_output
        
        x = x.permute(0,2,1)
        x = x.reshape(B,T,N,N)
        return x
        
    
class Model(nn.Module):
    def __init__(
        self,
        num_nodes,
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
        super(Model, self).__init__()
        self.num_nodes = num_nodes # N
        self.in_seq_len = in_seq_len # T
        self.out_seq_len = out_seq_len # T
        self.d_model = d_model # D
        self.heads = heads 
        self.num_encoder_layers = num_encoder_layers 
        self.s_in_channels = s_in_channels
        self.p_in_channels = p_in_channels
        self.t_in_channels = t_in_channels
        self.hidden_channels = hidden_channels
        self.attention_window = attention_window
        self.fc = nn.Linear(self.num_nodes,self.d_model)
        
        self.out = nn.Linear(self.d_model,self.num_nodes)
        self.x_r_fc = nn.Linear(self.num_nodes, self.d_model)
        self.vsa = VSA(self.d_model,self.heads,self.num_encoder_layers)
        self.bsta = BSTA(self.num_nodes,self.in_seq_len,self.out_seq_len,self.s_in_channels,self.p_in_channels,self.t_in_channels,self.hidden_channels,self.attention_window,self.d_model,self.heads,self.num_encoder_layers)
        self.odlinear = ODLinear(self.num_nodes,self.in_seq_len, self.out_seq_len)
        self.odta = ODTA(self.num_nodes,
        self.in_seq_len,
        self.out_seq_len,
        self.d_model,
        self.heads,
        self.num_encoder_layers)
        self.gcn = GCN(self.d_model,self.d_model,self.d_model,0.1)
        
    def forward(self, x, x_r, te, se, pe, adj, s_adj):
        # x = self.odgat(x,pe)
        # x_r = self.vsa(x_r,adj,s_adj)
        # x_r = self.fc(x)
        x_r = self.gcn(x_r,adj)
        # x_trans = x.transpose(-1,-2)
        # x = torch.matmul(x,x_trans)
        # x_r = self.x_r_fc(x_r)
        # x_r = torch.randn_like(x_r)
        x = self.bsta(x, x_r, te, se, pe, adj,s_adj)
        # x = self.odta(x_r)
        # x_od = self.odlinear(x, te, se, pe, adj,s_adj)
        # x = self.gat(x_r,adj)
        # x = self.out(x)
        return x