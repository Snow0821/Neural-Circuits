import torch
import torch.nn as nn
import torch.nn.functional as F
from .logic_gates import NAND, XOR

class btnnLayer(nn.Module):
    def __init__(self, *kwargs):
        super().__init__()
        self.pr = nn.Linear(*kwargs)
        self.nr = nn.Linear(*kwargs)

        torch.nn.init.uniform_(self.pr.weight, -1, 1)
        torch.nn.init.uniform_(self.nr.weight, -1, 1)

    def forward(self, x):
        pr = self.pr.weight
        nr = self.nr.weight
        qpr = (pr > 0).float()
        qnr = (nr > 0).float()

        # Apply quantization using posNet with detach
        qpr = qpr - pr.detach() + pr
        qnr = qnr - nr.detach() + nr

        # Compute linear transformations
        yr = F.linear(x, qpr) - F.linear(x, qnr)
        qyr = (yr > 0).float()
        ret = qyr - yr.detach() + yr

        return ret

class MonoLogicGateLayer(nn.Module):
    def __init__(self, x, y):
        super().__init__()
        self.linear = nn.Linear(x, y)
    
    def forward(self, x):
        w = (self.linear.weight > 0).float() - self.linear.weight.detach() + self.linear.weight

class NandOrPool(nn.Module):
    def __init__(self, x):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(x // 2 + x % 2, dtype=torch.float16))

    def forward(self, x):
        batch_size, input_dim = x.shape
        if input_dim % 2 == 1:
            x = torch.cat([x, torch.zeros(batch_size, 1, device=x.device)], dim=1)

        a, b = x[:, ::2], x[:, 1::2]

        alpha = (self.weights > 0).float() - self.weights.detach() + self.weights
        output = 1 - a * b + alpha * (1 + a + b)

        return output
