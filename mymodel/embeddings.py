import numpy as np
import torch
import torch.nn as nn
from scipy.sparse import coo_matrix
from torch_geometric.data import Data
from models import GAT

class ODGAT(torch.nn.Module):
  def __init__(self, in_channels, hidden_channels, out_channels, heads, dropout):
    super(ODGAT,self).__init__()
    self.gat = GAT(in_channels,hidden_channels,out_channels,heads,dropout)
  
  def forward(self, x, adj):
    return self._get_od_emb(x,adj)
  
  def _get_od_emb(self,x,adj):
    od_emb = []
    for i in range(adj.size(0)):
      a=torch.cuda.LongTensor(adj[i])
      ed = coo_matrix(a.cpu())
      edge_index = torch.cuda.LongTensor(np.array([ed.row, ed.col]))
      f = x[i]
      data = Data(x=f,edge_index=edge_index)
      output = self.gat(data.x, data.edge_index)
      od_emb.append(output)
    od_emb = torch.stack(od_emb,dim=0)
    return od_emb
  
class SPGAT(torch.nn.Module):
  def __init__(self,s_in_channels,p_in_channels,t_in_channels,o_in_channels,hidden_channels,out_channels,heads,dropout):
    super(SPGAT, self).__init__()
    
    self.tf = nn.Linear(t_in_channels,hidden_channels) # temporal fc layer
    self.sf = nn.Linear(s_in_channels,hidden_channels) # static embeddings fc layer
    self.pf = nn.Linear(p_in_channels,hidden_channels) # poi fc layer
    self.fc = nn.Linear(hidden_channels,out_channels) # fc layer
    
    # OD embedding
    self.od_emb = ODGAT(o_in_channels, hidden_channels, out_channels, heads, dropout)
    # out
    self.out = nn.Linear(out_channels*2,out_channels)
  
  def od_out(self,x,adj):
    od = []
    for i in range(x.size(0)):
      od_item = self.od_emb(x[i],adj[i])
      od.append(od_item)
    od = torch.stack(od,dim=0)
    return od
    
  def forward(self,x,adj,te,se,pe):
    # temporal embedding
    t = self.tf(te)
    t = self.fc(t)
    # static embedding
    s = self.sf(se)
    s = self.fc(s)
    # poi embedding
    p = self.pf(pe)
    p = self.fc(p)
    # fussion
    sp = torch.cat((s,p),dim=1)
    sp = self.out(sp)
    st = sp.unsqueeze(0).unsqueeze(0).expand(x.size(0), x.size(1), -1, -1) + t.unsqueeze(2).expand(-1, -1, x.size(2), -1)
    # OD embedding
    od = self.od_out(x,adj)
    o = torch.cat((od,st),dim=3)
    o = self.out(o)
    return o  
  