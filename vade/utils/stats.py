import torch

def diagonal_mvn_log_prob(x, mu, logvar, device):
    c = 2 * torch.pi * torch.ones(1).to(device)
    return (-0.5 * (torch.log(c) + logvar + (x - mu).pow(2).div(torch.exp(logvar)))).sum(dim=-1)