import torch

def NAND(a, b):
    return ~(a.int() & b.int()) & 1  # Bitwise NAND

def XOR(a, b):
    return (a.int() ^ b.int()) & 1  # Bitwise XNOR

def NOR(a, b):
    return ~(a | b) & 1  # Bitwise NOR
