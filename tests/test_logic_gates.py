import torch
from neural_circuits.logic_gates import NAND, XNOR, NOR

def test_NAND():
    a = torch.tensor([0, 0, 1, 1])
    b = torch.tensor([0, 1, 0, 1])
    expected = torch.tensor([1, 1, 1, 0])
    assert torch.equal(NAND(a, b), expected)

def test_XNOR():
    a = torch.tensor([0, 0, 1, 1])
    b = torch.tensor([0, 1, 0, 1])
    expected = torch.tensor([1, 0, 0, 1])
    assert torch.equal(XNOR(a, b), expected)

def test_NOR():
    a = torch.tensor([0, 0, 1, 1])
    b = torch.tensor([0, 1, 0, 1])
    expected = torch.tensor([1, 0, 0, 0])
    assert torch.equal(NOR(a, b), expected)