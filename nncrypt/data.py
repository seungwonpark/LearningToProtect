import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


def create_dataloader(hp, train):
    dataset = BitsDataset(hp, train)
    return DataLoader(dataset=dataset, batch_size=256, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)


class BitsDataset(Dataset):
    def __init__(self, hp, train):
        self.train = train
        self.plain_len = hp.data.plain
        self.key_len = hp.data.key
        self.steps = 1000 if self.train else 10

    def __len__(self):
        return 256 * self.steps

    def __getitem__(self, idx):
        plainE = np.random.randint(0, 2, size=(plain_len))
        keyE = np.random.randint(0, 2, size=(key_len))
        plainAB = np.random.randint(0, 2, size=(plain_len))
        keyAB = np.random.randint(0, 2, size=(key_len))

        if self.train:
            return plainE, keyE, plainAB, keyAB
        else:
            return plainE, keyE
