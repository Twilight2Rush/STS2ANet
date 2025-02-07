import torch
from torch.utils.data import Dataset, DataLoader

class ODDataset(Dataset):
    def __init__(self, x, adj, te, y, c, tc, ac):
        self.x = x
        self.adj = adj
        self.te = te
        self.y = y
        self.c = c
        self.tc = tc
        self.ac = ac

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        sample = {
            'x': self.x[idx],
            'te': self.te[idx],
            'adj': self.adj[idx],
            'c': self.c[idx],
            'tc': self.tc[idx],
            'ac': self.ac[idx],
        }
        if self.y is not None:
            sample['y'] = self.y[idx]
        return sample