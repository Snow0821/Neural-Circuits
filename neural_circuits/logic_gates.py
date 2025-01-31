import torch

def NAND(a, b):
    return ~(a & b) & 1  # Bitwise NAND

def XNOR(a, b):
    return ~(a ^ b) & 1  # Bitwise XNOR

def NOR(a, b):
    return ~(a | b) & 1  # Bitwise NOR
