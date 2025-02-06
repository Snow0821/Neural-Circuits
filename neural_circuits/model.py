import torch
from torch import nn
from .layers import NandOrPool, btnnLayer

class BinaryLogicMNIST(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(btnnLayer(28 * 28, 16, 4))
        for _ in range(4):
            self.layers.append(btnnLayer(16, 16, 4))
        self.layers.append(btnnLayer(16, 16, 1))
        self.layers.append(btnnLayer(16, 10))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
