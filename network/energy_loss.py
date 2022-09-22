import torch

def nll(energy, batch_size):
    energy_pos = energy[:, 0].view(batch_size, -1)
    partition_estimate = -1 * energy
    partition_estimate = torch.logsumexp(partition_estimate, dim=1, keepdim=True)
    # This line below is essentially NLL
    predL = energy_pos + partition_estimate
    predL = predL.mean()
    return predL


def mee(energy, batch_size, beta=0.8):
    energy_pos = energy[:, 0].view(batch_size, -1)
    energy_neg = energy[:, 1].view(batch_size, -1)
    # This line below is essentially Minimum Empirical Loss
    predL = 1 - (torch.exp(-1 * beta * energy_pos) / torch.sum(torch.exp(-1 * beta * energy)))
    predL = predL.mean()
    return predL

def square_exponential(energy, batch_size, gamma=2.2):
    energy_pos = energy[:, 0].view(batch_size, -1)
    energy_neg = energy[:, 1].view(batch_size, -1)
    # This is square Exponential
    predL = torch.pow(energy_pos, 2) + gamma * torch.exp(-1 * energy_neg)
    predL = predL.mean()
    return predL


def square_square(energy, batch_size, m=1):
    energy_pos = energy[:, 0].view(batch_size, -1)
    energy_neg = energy[:, 1].view(batch_size, -1)
    # This is square Exponential
    zeros = torch.zeros_like(energy_neg)
    predL = torch.pow(energy_pos, 2) + torch.pow(torch.max(zeros, m - (energy_neg)), 2)
    predL = predL.mean()
    return predL

def log_loss(energy, batch_size):
    energy_pos = energy[:, 0].view(batch_size, -1)
    energy_neg = energy[:, 1].view(batch_size, -1)
    # This is square Exponential
    predL = torch.log(1 + torch.exp(energy_pos - energy_neg))
    predL = predL.mean()
    return predL

def hinge_loss(energy, batch_size, m=1):
    energy_pos = energy[:, 0].view(batch_size, -1)
    energy_neg = energy[:, 1].view(batch_size, -1)
    # This is square Exponential
    predL = torch.max(torch.zeros_like(energy_pos), m + (energy_pos - energy_neg))
    predL = predL.mean()
    return predL
