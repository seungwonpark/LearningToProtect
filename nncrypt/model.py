import torch
import torch.nn as nn
import torch.nn.functional as F


class Alice(nn.Module):
    def __init__(self, hp):
        super(Alice, self).__init__()
        self.depth = hp.alice.depth
        self.hidden = hp.alice.hidden

        self.mlp = nn.ModuleList(
            [nn.Linear(hp.data.plain + hp.data.key, self.hidden)] +
            [nn.Linear(self.hidden, self.hidden) for _ in range(self.depth-1)])
        self.last = nn.Linear(self.hidden, hp.data.cipher)

    def forward(self, p, k):
        x = torch.cat((p, k), dim=-1)

        for layer in self.mlp:
            x = F.relu(layer(x))

        x = torch.tanh(self.last(x))
        return x


class Bob(nn.Module):
    def __init__(self, hp):
        super(Bob, self).__init__()
        self.depth = hp.bob.depth
        self.hidden = hp.bob.hidden

        self.mlp = nn.ModuleList(
            [nn.Linear(hp.data.cipher + hp.data.key, self.hidden)] +
            [nn.Linear(self.hidden, self.hidden) for _ in range(self.depth-1)])
        self.last = nn.Linear(self.hidden, hp.data.plain)

    def forward(self, c, k):
        x = torch.cat((c, k), dim=-1)

        for layer in self.mlp:
            x = F.relu(layer(x))

        x = torch.tanh(self.last(x))
        return x


class Eve(nn.Module):
    def __init__(self, hp):
        super(Eve, self).__init__()
        self.depth = hp.eve.depth
        self.hidden = hp.eve.hidden

        self.mlp = nn.ModuleList(
            [nn.Linear(hp.data.cipher, self.hidden)] + 
            [nn.Linear(self.hidden, self.hidden) for _ in range(self.depth-1)])
        self.last = nn.Linear(self.hidden, hp.data.plain)

    def forward(self, c):
        x = c

        for layer in self.mlp:
            x = F.relu(layer(x))

        x = torch.tanh(self.last(x))
        return x
