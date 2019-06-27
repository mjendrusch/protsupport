import torch
import torch.nn as nn
import torch.nn.functional as func

class GaussianRBF(nn.Module):
  def __init__(self, low, high, num):
    super(GaussianRBF, self).__init__()
    rbf_positions = torch.tensor(
      list(range(num)),
      dtype=torch.float
    )
    self.positions = nn.Parameter(rbf_positions / num * (high - low) + low)
    self.weight = nn.Parameter(num / (high - low))

  def forward(self, positions):
    exponent = -self.weight * (self.positions[None, :] - positions) ** 2
    return torch.exp(exponent)

def gaussian_rbf(inputs, low=0, high=20, num=16):
  rbf_positions = torch.tensor(
    list(range(num)),
    dtype=torch.float
  ).to(inputs.device)
  rbf_positions = rbf_positions / num * (high - low) + low
  weight = num / (high - low)
  exponent = -weight * (rbf_positions[None, :] - inputs) ** 2
  return torch.exp(exponent)
