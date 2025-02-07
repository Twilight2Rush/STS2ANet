import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import pickle
import dgl
import random
from torch.utils.data import DataLoader
from dataset import ODDataset
from sklearn.model_selection import train_test_split

# def is_cross_day_hour_tail(hour):
#   if hour >= 0 and hour <=6:
#     return True
#   else:
#     return False

# def is_cross_day_hour_head(hour):
#   if hour >= 18 and hour <=23:
#     return True
#   else:
#     return False

# def get_cross_day_emb(t):
#   weekends = [6,7]
#   cd1 = cd2 = cd3 = 0
#   if is_cross_day_hour_head(t[1]):
#     cd3 = 1
#     if t[0].weekday()+1 in weekends:
#       cd1 = 1
#       if t[0].weekday()+1 == 6:
#         cd2 = 1
#   elif is_cross_day_hour_tail(t[1]):
#     cd3 = 1
#     if t[0].weekday()+1 in weekends:
#       cd2 = 1
#       if t[0].weekday()+1 == 7:
#         cd1 = 1
#   return np.array([cd1,cd2,cd3])


def t_emb(t):
    d_w_list = range(1, 8)  # day of week
    d_w_oh_list = np.eye(len(d_w_list))  # one hot
    d_w_emb = d_w_oh_list[d_w_list.index(t[0].weekday() + 1)]  # day of week embedding
    h_d_list = range(0, 24)  # hour of day
    h_d_oh_list = np.eye(len(h_d_list))  # one hot
    h_d_emb = h_d_oh_list[h_d_list.index(t[1])]
    emb = np.hstack((d_w_emb, h_d_emb))
    return emb

def load_data(args):
    with open(args.data_path, "rb") as f:
        data = pickle.load(f)
    te = []
    x = []
    adj = []
    # t_label = []
    for t, v in data.items():
        te_item = torch.FloatTensor(t_emb(t))  # temporal embedding
        # t_label_item = t[0].weekday() + 1 # day of week
        x_item = torch.FloatTensor(v)  # features
        adj_item = x_item + torch.eye(x_item.size(0))
        adj_item = torch.log(adj_item+1)
        # adj_item = torch.FloatTensor(np.where(v != 0, 1, 0))  # adjacency  matrix
        te.append(te_item)
        x.append(x_item)
        adj.append(adj_item)
        # t_label.append(t_label_item)
    te = torch.stack(te, dim=0)
    x = torch.stack(x, dim=0)
    adj = torch.stack(adj, dim=0)
    # t_label = torch.Tensor(t_label)
    x, te, adj, y, c, tc, ac= split_data(x, te, adj, args)
    adj = adj.long()
    return x[:-6], te[:-6], adj[:-6], y[:-6], c[:-6], tc[:-6], ac[:-6]

def split_data(x, te, adj, args):
    te_m = []
    # t_label_m = []
    x_m = []
    adj_m = []
    y_m = []
    c_m = []
    tc_m = []
    ac_m = []
    x = x[6:]
    te = te[6:]
    adj = adj[6:]
    for i in range(
        ((len(x) - args.in_seq_len - args.out_seq_len) // args.slide_len) + 1
    ):
        sliced_te = te[i * args.slide_len : i * args.slide_len + args.in_seq_len + args.out_seq_len]
        # if t_label[i * args.slide_len+args.in_seq_len] in [1,2,3,4]:
        #     sliced_t_label = 0
        # elif t_label[i * args.slide_len+args.in_seq_len] == 5:
        #     sliced_t_label = 1
        # elif t_label[i * args.slide_len+args.in_seq_len] == 6:
        #     sliced_t_label = 2
        # else:
        #     sliced_t_label = 3    
        sliced_x = x[i * args.slide_len : i * args.slide_len + args.in_seq_len]
        sliced_adj = adj[i * args.slide_len : i * args.slide_len + args.in_seq_len]
        sliced_c = []
        sliced_tc = []
        sliced_ac = []
        for j in range(1,args.his_window+1):
            if i-7*j+1>0:
                c_item = x[(i-7*j) * args.slide_len + args.in_seq_len : (i-7*j) * args.slide_len + args.in_seq_len + args.out_seq_len]
                tc_item = te[(i-7*j) * args.slide_len + args.in_seq_len : (i-7*j) * args.slide_len + args.in_seq_len + args.out_seq_len]
                tc_item = tc_item.repeat(2,1)
                ac_item = adj[(i-7*j) * args.slide_len + args.in_seq_len : (i-7*j) * args.slide_len + args.in_seq_len + args.out_seq_len]
            else:
                c_item = torch.zeros_like(x[:args.in_seq_len])
                tc_item = torch.zeros_like(te[:args.in_seq_len + args.out_seq_len])
                ac_item = torch.zeros_like(adj[:args.in_seq_len])
            sliced_c.append(c_item)
            sliced_tc.append(tc_item)
            sliced_ac.append(ac_item)
        sliced_c = torch.stack(sliced_c,dim=0)
        sliced_tc = torch.stack(sliced_tc,dim=0)
        sliced_ac = torch.stack(sliced_ac,dim=0)
        y_item = x[
            i * args.slide_len
            + args.in_seq_len : i * args.slide_len
            + args.in_seq_len
            + args.out_seq_len
        ]
        te_m.append(sliced_te)
        # t_label_m.append(sliced_t_label)
        x_m.append(sliced_x)
        adj_m.append(sliced_adj)
        y_m.append(y_item)
        c_m.append(sliced_c)
        tc_m.append(sliced_tc)
        ac_m.append(sliced_ac)
    te_m = torch.stack(te_m, dim=0)
    x_m = torch.stack(x_m, dim=0)
    adj_m = torch.stack(adj_m, dim=0)
    y_m = torch.stack(y_m, dim=0)
    c_m = torch.stack(c_m, dim=0)
    tc_m = torch.stack(tc_m, dim=0)
    ac_m = torch.stack(ac_m, dim=0)
    return x_m, te_m, adj_m, y_m, c_m, tc_m, ac_m


def od_dataloader(args):
    x,  te, adj, y, c, tc, ac = load_data(args)
    x_train, x_val, adj_train, adj_val, te_train, te_val, y_train, y_val, c_train, c_val, tc_train, tc_val, ac_train, ac_val = train_test_split(x, adj, te, y, c, tc, ac, test_size=args.test_size, shuffle=False)
    x_val, x_test, adj_val, adj_test, te_val, te_test, y_val, y_test, c_val, c_test, tc_val, tc_test, ac_val, ac_test = train_test_split(x_val, adj_val, te_val, y_val, c_val, tc_val, ac_val, test_size=0.5, shuffle=False)
    train_dataset = ODDataset(x_train,  adj_train, te_train, y_train, c_train, tc_train, ac_train)
    val_dataset = ODDataset(x_val, adj_val, te_val,  y_val, c_val, tc_val, ac_val)
    test_dataset = ODDataset(x_test, adj_test, te_test,  y_test, c_test, tc_test, ac_test)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    print(f"train number:{len(x_train)}")
    print(f"validation number:{len(x_val)}")
    print(f"test number:{len(x_test)}")
    
    return train_dataloader,val_dataloader, test_dataloader

def attr_score(x):
  div = x.size(2)//2
  x_r = x[:, :, :div]
  x_c = x[:, :, div:]
  row = x_r / x_r.sum(dim=1,keepdim=True)
  col = x_c / x_c.sum(dim=1,keepdim=True)
  a = torch.transpose(col, 1, 2) - row
  a[torch.isnan(a)] = 0
  return a

def mask(x,a):
  m = torch.zeros_like(torch.randn(x.size(0),x.size(0),x.size(1),x.size(1)))
  for t in range(x.size(0)):
    for t_hat in range(x.size(0)):
      m[t][t_hat] = (t-t_hat)/x.size(0)*a[t_hat]
  m = m.transpose(1,2)
  m = m.reshape(m.size(0)*m.size(1),m.size(2)*m.size(3))
  m = torch.clamp(m,min=0)
  return m 


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False


def masked_mse(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (preds-labels)**2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def masked_rmse(preds, labels, null_val=np.nan):
    return torch.sqrt(masked_mse(preds=preds, labels=labels, null_val=null_val))

def masked_mae(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def masked_mape(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    
    loss = torch.abs(preds-labels)/(labels+0.1)
    
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)

    return torch.mean(loss)

def ODLoss(pred,real):
    I1_y = torch.where(real==0,1,0)
    I2_y = torch.where(pred>0,1,0)
    loss = (I1_y * I2_y + 1 - I1_y)*((real-pred)**2)
    return torch.mean(loss)

def metric(pred, real):
    mse = masked_mse(pred,real).item()
    mae = masked_mae(pred,real).item()
    mape = masked_mape(pred,real).item()
    rmse = masked_rmse(pred,real).item()
    return mse,rmse,mae,mape

def generate_contrastive_triplets(x, t_label):
    triplets = []
    for i in range(x.size(0)):
        anchor_x = x[i]
        anchor_t_label = t_label[i]
        # positive smple
        positive_indices = [j for j in range(x.size(0)) if t_label[j] == anchor_t_label and j != i]
        if not positive_indices:
            continue
        positive_index = random.choice(positive_indices)
        positive_x = x[positive_index]
        # negative sample
        negative_indices = [j for j in range(x.size(0)) if t_label[j] != anchor_t_label]
        if not negative_indices:
            continue
        negative_index = random.choice(negative_indices)
        negative_x = x[negative_index]
        triplets.append((anchor_x, positive_x, negative_x))
    return triplets
