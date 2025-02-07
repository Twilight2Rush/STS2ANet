import torch
import os
import numpy as np
from utils import metric, generate_contrastive_triplets, masked_mse
import pandas as pd


def train(
    model,
    train_dataloader,
    se,
    pe,
    valid_dataloader,
    optimizer,
    criterion,
    contrastive_loss,
    scheduler,
    args
):
    train_losses = []
    valid_losses = []
    best_result = float("inf")
    early_stopping_counter = 0
    model = model.to(args.device)
    model.train()
    for epoch in range(args.epoch):
        total_loss = 0
        total_len = 0
        for batch in train_dataloader:
            x = batch["x"].to(args.device)
            B,T,N,N = x.size()
            x_r  = torch.randn(N,N)
            x_r = x_r.repeat(B,T,1,1).to(args.device)
            # x_r = torch.randn_like(x)
            adj = batch["adj"].to(args.device)
            s_adj = torch.cuda.LongTensor(pd.read_csv('../data/processed data/adj.csv',index_col= 0).values)
            te = batch["te"].to(args.device)
            y = batch["y"].to(args.device)
            if args.model_name == 'wsta':
                c = batch["c"].to(args.device) 
                tc = batch["tc"].to(args.device)
                ac = batch["ac"].to(args.device)
                p = model(x, x_r, te, se, pe, adj, c, tc, ac, s_adj)
            else:
                p = model(x, x_r, te, se, pe, adj, s_adj)
            optimizer.zero_grad()
            odloss = criterion(p, y)
            # triplets = generate_contrastive_triplets(p, t_label)
            # con_loss = contrastive_loss(triplets)
            con_loss = 0
            # print(f"train odloss:{odloss}")
            # print(f"train con_loss:{con_loss}")
            loss = odloss + con_loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)
            total_len += x.size(0)
            del x,adj,te,y
        average_loss = total_loss / total_len
        train_losses.append(average_loss)
        print("Epoch: {:02d}, Loss: {:.4f}".format(epoch, average_loss))
        test_loss, p_total = test(model, valid_dataloader, se, pe, criterion, contrastive_loss,args)
        model.train()
        valid_losses.append(test_loss)
        if test_loss < best_result:
            best_result = test_loss
            print("test loss: {:.4f}".format(test_loss))
            parent_folder = os.path.dirname(args.model_path)
            if not os.path.exists(parent_folder):
                os.makedirs(parent_folder)
            torch.save(
                    model.state_dict(),
                    f"{args.model_path}/model_{args.model_name}_seed_{args.seed}__epoch_{args.epoch}_batchsize_{args.batch_size}_lr_{args.lr}_layer_{args.num_encoder_layers}_window_{args.attention_window}.pth",
                )
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
        if early_stopping_counter >= args.patience:
            print(f"Early stopping at epoch {epoch}.")
            break
        scheduler.step()
    np.save(
            f"{args.loss_path}/train_loss/seed_{args.seed}_epoch_{args.epoch}_batchsize_{args.batch_size}_lr_{args.lr}_layer_{args.num_encoder_layers}_window_{args.attention_window}_model_{args.model_name}",
            np.array(train_losses),
        )
    np.save(
            f"{args.loss_path}/valid_loss/seed_{args.seed}_epoch_{args.epoch}_batchsize_{args.batch_size}_lr_{args.lr}_layer_{args.num_encoder_layers}_window_{args.attention_window}_model_{args.model_name}",
            np.array(valid_losses),
        )


def test(model, dataloader, se, pe, criterion, contrastive_loss, args, ispred=False):
    model = model.to(args.device)
    with torch.no_grad():
        model.eval()
        total_loss = 0
        total_len = 0
        p_total = []
        y_total = []
        for batch in dataloader:
            x = batch["x"].to(args.device)
            B,T,N,N = x.size()
            x_r  = torch.randn(N,N)
            x_r = x_r.repeat(B,T,1,1).to(args.device)
            # x_r = torch.randn_like(x)
            adj = batch["adj"].to(args.device)
            s_adj = torch.cuda.LongTensor(pd.read_csv('../data/processed data/adj.csv',index_col= 0).values)
            te = batch["te"].to(args.device)
            y = batch["y"].to(args.device)
            if args.model_name == 'wsta':
                c = batch["c"].to(args.device) 
                tc = batch["tc"].to(args.device)
                ac = batch["ac"].to(args.device)
                p = model(x, x_r, te, se, pe, adj, c, tc, ac, s_adj)
            else:
                p = model(x, x_r, te, se, pe, adj, s_adj)
            odloss = criterion(p, y)
            # triplets = generate_contrastive_triplets(p, t_label)
            # con_loss = contrastive_loss(triplets)
            con_loss = 0
            # print(f"test odloss:{odloss}")
            # print(f"test con_loss:{con_loss}")
            loss = odloss + con_loss
            total_loss += loss.item() * x.size(0)
            total_len += x.size(0)
            p_total.append(p)
            y_total.append(y)
            del x,adj,te,y
        mean_loss = total_loss / total_len
        p_total = torch.cat(p_total, dim=0)
        y_total = torch.cat(y_total, dim=0)
        if ispred:
            p_total = torch.where(p_total<0,0,p_total)
            mse,rmse,mae,mape = metric(p_total,y_total)
            return mse,rmse,mae,mape,p_total,y_total
        else:
            return mean_loss, p_total
