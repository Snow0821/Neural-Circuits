import torch
import torch.nn as nn
from .logic_gates import NAND, XNOR, NOR

class BinaryLogicLayer(nn.Module):
    def __init__(self, in_features, out_features, logic_gate=NAND):
        """
        Binary Neural Network Layer using custom logic gates.
        
        :param in_features: Number of input features
        :param out_features: Number of output features
        :param logic_gate: Logic gate function (NAND, XNOR, NOR, etc.)
        """
        super().__init__()
        self.logic_gate = logic_gate
        self.weights = nn.Parameter(torch.randint(0, 2, (in_features, out_features)).float(), requires_grad=True)
        self.bias = nn.Parameter(torch.randint(0, 2, (out_features,)).float(), requires_grad=True)

    def forward(self, x):
        """
        Forward pass using the selected logic gate.
        """
        return self.logic_gate(x @ self.weights, self.bias)

# Predefined layers using different logic gates
class BinaryNANDLayer(BinaryLogicLayer):
    def __init__(self, in_features, out_features):
        super().__init__(in_features, out_features, logic_gate=NAND)

class BinaryXNORLayer(BinaryLogicLayer):
    def __init__(self, in_features, out_features):
        super().__init__(in_features, out_features, logic_gate=XNOR)

class BinaryNORLayer(BinaryLogicLayer):
    def __init__(self, in_features, out_features):
        super().__init__(in_features, out_features, logic_gate=NOR)
