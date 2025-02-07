import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from utils import od_dataloader,ODLoss,set_seed,masked_mse,masked_mae
from config import get_configs
import numpy as np
import pandas as pd
from models import VSTA, ContrastiveLoss, BSTA, ODTA,ESA, WSTA, Model
from lib import train, test

args = get_configs()
args.model_name = 'dlsta'

# seeds = [42, 3407, 114514]
seeds = [42]

se = torch.Tensor(np.load(args.se_path)).to(args.device)
pe = torch.Tensor(pd.read_csv(args.pe_path,index_col=0).values).to(args.device)
train_dataloader, val_dataloader, test_dataloader = od_dataloader(args)

for seed in seeds:
    args.seed = seed
    set_seed(args.seed)
    print("*" * 10)
    print(f"start seed {args.seed}")
    print("*" * 10)
    model = Model(args.num_nodes,
        args.in_seq_len,
        args.out_seq_len,
        args.s_feature,
        args.p_feature,
        args.t_feature,
        args.hidden_channels,
        args.attention_window,
        args.d_model,
        args.heads,
        args.num_encoder_layers)
    # model = ESA(1,
    #     16,
    #     args.in_seq_len,
    #     args.num_encoder_layers
    #             )
    # model = META(args.heads,args.d_model,args.in_seq_len,args.num_nodes,args.num_encoder_layers)
    # model = META_CD(args.heads,args.d_model,args.in_seq_len,args.num_nodes,args.num_encoder_layers)
    # model = META_CDW(args.heads,args.d_model,args.in_seq_len,args.num_nodes,args.num_encoder_layers, args.his_window)
    # model = MVSTA(args.heads,args.in_seq_len,66,args.n_feature, args.d_model,args.s_feature,args.p_feature,args.t_feature, args.num_encoder_layers)
    # model = SPGAT(
    #         args.num_nodes,
    #         args.hidden_channels,
    #         args.d_model,
    #         args.heads,
    #         args.dropout,
    #     )
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = ODLoss
    contrastive_loss = ContrastiveLoss(margin=5)
    scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    train(model,train_dataloader,se,pe,val_dataloader,optimizer,criterion,contrastive_loss,scheduler,args)
    torch.cuda.empty_cache()