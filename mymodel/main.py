import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from config import get_configs
from utils import load_data, load_data_week, set_seed, metric, masked_mse
from dataset import ODDataset, ODWEEKDataset
from models import MYMODEL, MYMODEL2, MYMODEL3, MYMODEL4
from lib import train, test

args = get_configs()
seeds = [42, 0, 123, 29, 66]


args.model_name = "cross_day_week"  # cross_day
args.alpha = 0.4
if args.model_name == 'cross_day_week':
    x, c, te, tc, adj, ac, y = load_data_week(args)
else:    
    te, x, adj, y = load_data(args)
for seed in seeds:
    args.seed = seed
    set_seed(args.seed)
    print("*" * 10)
    print(f"start seed {args.seed}")
    print("*" * 10)
    if args.model_name == "cross_day_week":
        (
            x_train,
            x_valid,
            c_train,
            c_valid,
            adj_train,
            adj_valid,
            ac_train,
            ac_valid,
            te_train,
            te_valid,
            tc_train,
            tc_valid,
            y_train,
            y_valid,
        ) = train_test_split(
            x, c, adj, ac, te, tc, y, test_size=args.test_size, random_state=args.seed
        )
        (
            x_train,
            x_test,
            c_train,
            c_test,
            adj_train,
            adj_test,
            ac_train,
            ac_test,
            te_train,
            te_test,
            tc_train,
            tc_test,
            y_train,
            y_test,
        ) = train_test_split(
            x_train,
            c_train,
            adj_train,
            ac_train,
            te_train,
            tc_train,
            y_train,
            test_size=args.test_size,
            random_state=args.seed,
        )

        train_dataset = ODWEEKDataset(
            x_train, c_train, adj_train, ac_train, te_train, tc_train, y_train
        )
        valid_dataset = ODWEEKDataset(
            x_valid, c_valid, adj_valid, ac_valid, te_valid, tc_valid, y_valid
        )
        test_dataset = ODWEEKDataset(
            x_test, c_test, adj_test, ac_test, te_test, tc_test, y_test
        )

        train_dataloader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=False
        )
        valid_dataloader = DataLoader(
            valid_dataset, batch_size=args.batch_size, shuffle=False
        )
        test_dataloader = DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=False
        )
    else:
        (
            x_train,
            x_valid,
            adj_train,
            adj_valid,
            te_train,
            te_valid,
            y_train,
            y_valid,
        ) = train_test_split(
            x, adj, te, y, test_size=args.test_size, random_state=args.seed
        )
        (
            x_train,
            x_test,
            adj_train,
            adj_test,
            te_train,
            te_test,
            y_train,
            y_test,
        ) = train_test_split(
            x_train,
            adj_train,
            te_train,
            y_train,
            test_size=args.test_size,
            random_state=args.seed,
        )

        train_dataset = ODDataset(x_train, adj_train, te_train, y_train)
        valid_dataset = ODDataset(x_valid, adj_valid, te_valid, y_valid)
        test_dataset = ODDataset(x_test, adj_test, te_test, y_test)

        train_dataloader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=False
        )
        valid_dataloader = DataLoader(
            valid_dataset, batch_size=args.batch_size, shuffle=False
        )
        test_dataloader = DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=False
        )

    se = torch.Tensor(np.load(args.se_path)).to(args.device)
    pe = torch.Tensor(pd.read_csv(args.pe_path, index_col=0).values).to(args.device)

    if args.model_name == "vanilla":
        model = MYMODEL(
            s_in_channels=se.size(1),
            p_in_channels=pe.size(1),
            t_in_channels=te.size(2),
            o_in_channels=x.size(3),
            num_nodes=x.size(2),
            in_seq_len=args.in_seq_len,
            out_seq_len=args.out_seq_len,
            num_encoder_layers=args.num_encoder_layers,
            d_model=args.d_model,
            hidden_channels=args.hidden_channels,
            heads=args.heads,
            dropout=args.dropout,
        ).to(args.device)
    elif args.model_name == "cross_day":
        model = MYMODEL2(
            s_in_channels=se.size(1),
            p_in_channels=pe.size(1),
            t_in_channels=te.size(2),
            o_in_channels=x.size(3),
            num_nodes=x.size(2),
            in_seq_len=args.in_seq_len,
            out_seq_len=args.out_seq_len,
            num_encoder_layers=args.num_encoder_layers,
            d_model=args.d_model,
            hidden_channels=args.hidden_channels,
            heads=args.heads,
            dropout=args.dropout,
            alpha=args.alpha,
        ).to(args.device)
    elif args.model_name == "cross_day_att":
        model = MYMODEL3(
            s_in_channels=se.size(1),
            p_in_channels=pe.size(1),
            t_in_channels=te.size(2),
            o_in_channels=x.size(3),
            num_nodes=x.size(2),
            in_seq_len=args.in_seq_len,
            out_seq_len=args.out_seq_len,
            num_encoder_layers=args.num_encoder_layers,
            d_model=args.d_model,
            hidden_channels=args.hidden_channels,
            heads=args.heads,
            dropout=args.dropout,
        ).to(args.device)
    elif args.model_name == "cross_day_week":
        model = MYMODEL4(
            s_in_channels=se.size(1),
            p_in_channels=pe.size(1),
            t_in_channels=te.size(2),
            o_in_channels=x.size(3),
            num_nodes=x.size(2),
            in_seq_len=args.in_seq_len,
            out_seq_len=args.out_seq_len,
            num_encoder_layers=args.num_encoder_layers,
            d_model=args.d_model,
            hidden_channels=args.hidden_channels,
            heads=args.heads,
            dropout=args.dropout,
            alpha = args.alpha
        ).to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = masked_mse
    scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    train(
        model,
        train_dataloader,
        se,
        pe,
        valid_dataloader,
        optimizer,
        criterion,
        scheduler,
        args,
    )
