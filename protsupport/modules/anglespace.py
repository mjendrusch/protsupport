import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as func

from torchsupport.structured import scatter
from torchsupport import structured as ts
from torchsupport.modules.activations.geometry import torus

class AngleProject(nn.Module):
  def __init__(self, in_size, angle_size):
    super(AngleProject, self).__init__()
    self.sin = nn.Linear(in_size, angle_size)
    self.cos = nn.Linear(in_size, angle_size)

  def forward(self, inputs):
    sin = self.sin(inputs)
    cos = self.cos(inputs)
    return torus(sin, cos)

class AngleLookup(nn.Module):
  def __init__(self, in_size, angle_size):
    super(AngleLookup, self).__init__()
    self.angle_size = angle_size
    self.register_parameter(
      "angles",
      nn.Parameter((torch.rand(angle_size, 3) - 0.5) * 2 * np.pi)
    )
    self.lookup = nn.Linear(in_size, angle_size)

  def forward(self, inputs):
    key = self.lookup(inputs).softmax(dim=1)
    the_angles = torch.atan2(key @ self.angles.sin(), key @ self.angles.cos())
    return the_angles

class AngleSample(nn.Module):
  def __init__(self, in_size, angle_size):
    super(AngleSample, self).__init__()
    self.angle_size = angle_size
    self.register_parameter(
      "angles",
      nn.Parameter((torch.rand(angle_size, 3) - 0.5) * 2 * np.pi)
    )
    self.lookup = nn.Linear(in_size, angle_size)

  def forward(self, inputs):
    key = self.lookup(inputs).softmax(dim=1)
    key = torch.distributions.RelaxedOneHotCategorical(0.1, key).rsample()
    the_angles = torch.atan2(key @ self.angles.sin(), key @ self.angles.cos())
    return the_angles

class AngleLookupGrid(nn.Module):
  def __init__(self, in_size, angle_size):
    super(AngleLookupGrid, self).__init__()
    self.angle_size = angle_size
    self.register_parameter(
      "angles",
      nn.Parameter((torch.rand(angle_size, 3) - 0.5) * 2 * np.pi)
    )
    self.lookup = nn.Linear(in_size, angle_size)

  def forward(self, inputs):
    key = self.lookup(inputs).softmax(dim=1)
    the_angles = torch.atan2(key @ self.angles.sin(), key @ self.angles.cos())
    return the_angles

class PositionLookup(nn.Module):
  def __init__(self, fragment_size=5):
    super(PositionLookup, self).__init__()
    self.register_buffer("bond_lengths", torch.tensor([1.46, 1.53, 1.33]).unsqueeze(0))
    self.register_buffer("bond_angles", np.pi - 1 / 180 * np.pi * torch.tensor([122.2, 111.9, 116.2]).unsqueeze(0))
    self.fragment_size = fragment_size

  def to_snr(self, torsions):
    angle_sin = self.bond_lengths * self.bond_angles.sin()
    angle_cos = self.bond_lengths * self.bond_angles.cos()
    x = angle_cos.unsqueeze(1)
    y = (torsions.cos() * angle_sin).unsqueeze(1)
    z = (torsions.sin() * angle_sin).unsqueeze(1)
    x = x.expand_as(y).contiguous()
    points = torch.cat((x, y, z), dim=1)
    points = points.permute(0, 2, 1)
    return points

  def fragment(self, inputs, indices):
    unique, count = indices.unique(return_counts=True)

    # compute padding accessor into fragments:
    fragment_pad = ((count + self.fragment_size - 1) // self.fragment_size) * self.fragment_size
    last_fragment_pad = fragment_pad - count
    fragment_offset = last_fragment_pad.roll(1)
    fragment_offset[0] = 0
    fragment_offset = torch.repeat_interleave(fragment_offset, count)
    fragment_access = torch.arange(count.sum()) + fragment_offset

    # pad fragments:
    result = torch.zeros(
      fragment_pad.sum(),
      *inputs.shape[1:]
    ).to(inputs.device)
    result[fragment_access] = inputs
    result = result.view(
      result.size(0) // self.fragment_size,
      self.fragment_size,
      *result.shape[1:]
    )

    return result, fragment_access

  def rotation(self, pos, ms):
    m_hat = ms[:, -1] / (ms[:, -1].norm(dim=1, keepdim=True) + 1e-16)
    n = torch.cross(ms[:, -2], m_hat)
    n_hat = n / (n.norm(dim=1, keepdim=True) + 1e-16)
    cross = torch.cross(n_hat, m_hat)
    rot = torch.cat((
      m_hat.unsqueeze(1),
      cross.unsqueeze(1),
      n_hat.unsqueeze(1)
    ), dim=1)
    rot = rot.squeeze(-1)
    rot = rot.permute(0, 2, 1)

    return rot

  def move(self, idx, position, pos_cache, ms_cache):
    rot = self.rotation(pos_cache, ms_cache)
    offset = pos_cache[:, -1]
    new_position = rot @ position[:, idx] + offset
    pos_cache = pos_cache.roll(-1, 1)
    pos_cache[:, -1] = new_position[:, :, -1:]
    ms_cache = pos_cache[:, 1:] - pos_cache[:, :-1]
    return new_position, pos_cache, ms_cache

  def init_cache(self, target, starter=None):
    # position cache:
    if starter is None:
      pos = torch.tensor([
        [-np.sqrt(0.5), np.sqrt(1.5), 0.0],
        [-np.sqrt(2),   0.0,          0.0],
        [0.0,           0.0,          0.0]
      ]).to(target.device)
    else:
      pos = starter
    pos = pos.unsqueeze(0).expand(
      target.size(0), *pos.shape
    ).contiguous()
    pos = pos.unsqueeze(-1) # make this a proper vector

    # m vector cache:
    ms = pos[:, 1:] - pos[:, :-1]
    return pos, ms

  def position(self, fragmented):
    atomized = fragmented.view(
      fragmented.size(0),
      -1,
      *fragmented.shape[3:]
    ).unsqueeze(-1)

    pos, ms = self.init_cache(atomized)

    fragment_result = torch.zeros_like(atomized)
    # compute fragment positions:
    for idx in range(self.fragment_size * 3):
      new_pos, pos, ms = self.move(idx, atomized, pos, ms)
      fragment_result[:, idx] = new_pos
    fragment_result = fragment_result.squeeze(-1).unsqueeze(0)
    fragment_result = fragment_result.permute(0, 1, 3, 2)

    result = torch.zeros_like(fragment_result)
    # align fragments:
    current = fragment_result[:, -1:]
    for idx in reversed(range(fragment_result.size(1) - 1)):
      pos, ms = self.init_cache(current, starter=fragment_result[0, idx, :, -3:].t())
      new_pos, pos, ms = self.move(0, current, pos, ms)
      current = torch.cat((fragment_result[:, idx:idx+1], new_pos.unsqueeze(0)), dim=-1)

    result = current.squeeze().t().contiguous()
    result = result - result[:1] # shift first atom to (0,0,0)
    result = result.view(-1, *fragmented.shape[2:])
    return result

  def stitch(self, fragmented_result, fragment_indices, indices):
    result = torch.zeros(indices.size(0), *fragmented_result.shape[1:]).to(fragmented_result.device)
    result = fragmented_result[fragment_indices]
    return result

  def forward(self, inputs, indices):
    inputs = self.to_snr(inputs)
    fragmented, fragment_indices = self.fragment(inputs, indices)
    positions = self.position(fragmented)
    result = self.stitch(positions, fragment_indices, indices)
    return result, indices

class LocalPositionLookup(PositionLookup):
  def init_local(self, angles, previous, idx):
    # position cache:
    if idx == 0:
      pos = torch.tensor([
        [-np.sqrt(0.5), np.sqrt(1.5), 0.0],
        [-np.sqrt(2),   0.0,          0.0],
        [0.0,           0.0,          0.0]
      ]).to(angles.device)
      pos = pos.unsqueeze(0).expand(
        angles.size(0), *pos.shape
      ).contiguous()
    else:
      pos = previous[:, idx - 1].clone()
    pos = pos.unsqueeze(-1) # make this a proper vector
    # m vector cache:
    ms = pos[:, 1:] - pos[:, :-1]
    return pos, ms

  def forward(self, angles, previous, idx):
    snr = self.to_snr(angles)
    pos, ms = self.init_local(angles, previous, idx)
    positions = []
    for idy in range(3):
      new_position, pos, ms = self.move(
        idy, snr.unsqueeze(-1), pos, ms
      )
      positions.append(new_position.unsqueeze(1).squeeze(dim=-1))
    positions = torch.cat(positions, dim=1)
    if idx == 0:
      positions = positions - positions[:, :1]
    return positions

class DistanceLookup(PositionLookup):
  def forward(self, inputs, indices):
    positions, indices = super().forward(inputs, indices)
    op = lambda x, y: (x - y).norm(dim=1)
    positions = positions.view(-1, 3)
    indices = torch.repeat_interleave(indices, 3)
    result, batch_indices = scatter.pairwise_no_pad(op, positions, indices)
    return result, batch_indices
