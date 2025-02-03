import torch
from torch import nn
from .layers import BinaryLogicGateLayer

class BinaryLogicMNIST(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer1 = BinaryLogicGateLayer(784)
        self.layer2 = BinaryLogicGateLayer(392)
        self.layer3 = BinaryLogicGateLayer(196)
        self.layer4 = BinaryLogicGateLayer(98)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x[:, :10]
