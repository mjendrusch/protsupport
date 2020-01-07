import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as func

from torchsupport.structured import scatter

class RGNLoss(nn.Module):
  def forward(self, inputs, target):
    target, mask, structure = target
    mask = mask.view(-1)
    target = target.view(-1, 3, 3)[:, 1]
    inputs = inputs.view(-1, 3, 3)[:, 1]
    dst = lambda x, y: (x - y).norm(dim=1)
    indices = structure.indices
    indices = indices[(mask > 0).nonzero().view(-1)]
    target = target[(mask > 0).nonzero().view(-1)]
    inputs = inputs[(mask > 0).nonzero().view(-1)]
    target, rmsd_indices = scatter.pairwise_no_pad(dst, target, indices)
    inputs, _ = scatter.pairwise_no_pad(dst, inputs, indices)
    result = (inputs - target) ** 2
    _, counts = indices.unique(return_counts=True)
    denominator = (counts * (counts - 1)).float().to(result.device)
    result = torch.sqrt(2 * scatter.add(result, rmsd_indices.to(result.device)) + 1e-6)
    result = result / torch.sqrt(denominator)
    result = result / counts.float().to(result.device)
    return result.mean()

class StochasticFullRGNLoss(nn.Module):
  def __init__(self, samples, relative=False):
    super().__init__()
    self.samples = samples
    self.relative = relative

  def positions(self, indices):
    _, counts = indices.unique(return_counts=True)
    result = []
    total = 0
    for count in counts:
      result.append(torch.randint(total, total + count, (1, self.samples)))
      total += count
    result = torch.cat(result, dim=0)
    return result

  def forward(self, inputs, target):
    target, mask, structure = target
    mask = mask.view(-1)
    target = target.view(-1, 3, 3)[:, 1]
    inputs = inputs.view(-1, 3, 3)[:, 1]
    indices = structure.indices
    indices = indices[(mask > 0).nonzero().view(-1)]
    target = target[(mask > 0).nonzero().view(-1)]
    inputs = inputs[(mask > 0).nonzero().view(-1)]
    positions = self.positions(indices)
    inputs = inputs[positions]
    target = target[positions]
    print("LESHAPE", inputs.shape, target.shape)
    in_distance = (inputs[:, None, :, :] - inputs[:, :, None, :]).norm(dim=-1)
    target_distance = (target[:, None, :, :] - target[:, :, None, :]).norm(dim=-1)
    result = (in_distance - target_distance)
    if self.relative:
      denominator = target_distance + (target_distance == 0).float()
      result = result / denominator
    result = result ** 2
    return result.mean()

class StochasticRGNLoss(nn.Module):
  def __init__(self, samples, relative=False):
    super().__init__()
    self.samples = samples
    self.relative = relative

  def pairs(self, indices):
    _, counts = indices.unique(return_counts=True)
    result = []
    total = 0
    for count in counts:
      result.append(torch.randint(total, total + count, (self.samples, 2)))
      total += count
    result = torch.cat(result, dim=0)
    return result[:, 0], result[:, 1]

  def forward(self, inputs, target):
    target, mask, structure = target
    mask = mask.view(-1)
    target = target.view(-1, 3, 3)[:, 1]
    inputs = inputs.view(-1, 3, 3)[:, 1]
    indices = structure.indices
    indices = indices[(mask > 0).nonzero().view(-1)]
    target = target[(mask > 0).nonzero().view(-1)]
    inputs = inputs[(mask > 0).nonzero().view(-1)]
    left, right = self.pairs(indices)
    in_distance = (inputs[left] - inputs[right]).norm(dim=1)
    target_distance = (target[left] - target[right]).norm(dim=1)
    result = (in_distance - target_distance)
    if self.relative:
      denominator = target_distance + (target_distance == 0).float()
      result = result / denominator
    result = result ** 2
    return result.mean()

class WeightedAngleLoss(nn.Module):
  def __init__(self, bins):
    super().__init__()
    self.bins = bins

    offset = 2 * np.pi / bins
    self.bin_position = [-np.pi + offset / 2 + idx * offset for idx in range(bins)]
    self.bin_position = torch.tensor(self.bin_position, dtype=torch.float)
    self.bin_position = self.bin_position.view(1, -1)

  def bin_angle(self, angle):
    distance = (angle - self.bin_position).norm(dim=1)
    bin_id = distance.argmax(dim=1).view(-1)
    return bin_id

  def bin_omega(self, angle):
    return (abs(angle) > np.pi / 2).long()

  def count(self, phi, psi, omega):
    indices = torch.cat([phi, psi, omega], dim=0)
    unique, count = indices.unique(dim=1, return_counts=True)
    counts = torch.zeros(self.bins, self.bins, 2)
    counts[unique[0], unique[1], unique[2]] += count
    return counts

  def forward(self, inputs, targets):
    if isinstance(targets, (list, tuple)):
      targets, mask = targets
      inputs = inputs[mask.nonzero().view(-1)]
      targets = targets[mask.nonzero().view(-1)]
    phi_bin = self.bin_angle(targets[:, 0])
    psi_bin = self.bin_angle(targets[:, 1])
    omega_bin = self.bin_omega(targets[:, 2])
    counts = self.count(phi_bin, psi_bin, omega_bin)
    target_counts = counts[phi_bin, psi_bin, omega_bin]
    result_sin = (inputs.sin() - targets.sin()).norm(dim=1)
    result_cos = (inputs.cos() - targets.cos()).norm(dim=1)
    result = (result_sin + result_cos) / target_counts
    return result.mean()
