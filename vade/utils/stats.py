import torch

def diagonal_mvn_log_prob(x, mu, var, device):
    c = 2 * torch.pi * torch.ones(1).to(device)
    return (-0.5 * (torch.log(c) + var.log() + (x - mu).pow(2).div(var))).sum(dim=-1)