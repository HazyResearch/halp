import torch

def single_to_half_det(tensor):
    return tensor.half().cuda()

def single_to_half_stoc(tensor):
    pass