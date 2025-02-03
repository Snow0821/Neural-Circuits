import torch
import torch.nn as nn
import torch.nn.functional as F
from .logic_gates import NAND, XOR

class BinaryLogicGateLayer(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size
        self.output_size = input_size // 2
        
        # Learnable weights (fp16)
        self.weights = nn.Parameter(torch.randn(self.output_size, dtype=torch.float16))

    def forward(self, x):
        x1, x2 = x[:, ::2], x[:, 1::2]

        xor_result = XOR(x1, x2)
        nand_result = NAND(x1, x2)

        gate_selection = (self.weights > 0).float().detach() + self.weights - self.weights.detach()
        output = gate_selection * xor_result + (1 - gate_selection) * nand_result

        return output