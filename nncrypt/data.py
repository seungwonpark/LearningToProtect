import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


def create_dataloader(hp, train):
    dataset = BitsDataset(hp, train)
    return DataLoader(dataset=dataset,
            batch_size=hp.train.batch_size,
            shuffle=True,
            num_workers=hp.train.num_workers,
            pin_memory=True,
            drop_last=True)


class BitsDataset(Dataset):
    def __init__(self, hp, train):
        self.hp = hp
        self.train = train
        self.plain = hp.data.plain
        self.key = hp.data.key
        self.steps = self.hp.train.steps[0 if self.train else 1]

    def __len__(self):
        return self.hp.train.batch_size * self.steps

    def __getitem__(self, idx):
        plainE = self.rand(self.plain)
        keyE = self.rand(self.key)
        plainAB = self.rand(self.plain)
        keyAB = self.rand(self.key)

        if self.train:
            return plainE, keyE, plainAB, keyAB
        else:
            return plainE, keyE
    
    def rand(self, size):
        x = np.random.randint(0, 2, size=size)
        x = x.astype(np.float32)
        x = torch.from_numpy(x)
        return x

