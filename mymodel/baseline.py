import torch


def HA(x):
    B,T,N,N = x.shape
    for t in range(T):
      x = torch.cat((x, torch.unsqueeze(torch.mean(x[:,t:t+T],dim=1), dim=1)), dim=1)
    p = x[:,24:48,:,:66]
    return p





