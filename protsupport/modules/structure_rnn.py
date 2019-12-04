import numpy as np

from scipy.spatial.ckdtree import cKDTree

import torch
import torch.nn as nn
import torch.nn.functional as func

from torchsupport.modules.residual import ResNetBlock1d
from torchsupport.structured import scatter, MILAttention, FullyConnectedScatter

from protsupport.modules.anglespace import AngleLookup, PositionLookup, LocalPositionLookup, DistanceLookup, AngleSample, AngleProject

class StructureRNN(nn.Module):
  def __init__(self, in_size, distance_size=64, context_size=800,
               hidden_size=800, radius=8, attention_size=128, heads=8):
    super().__init__()
    self.cell = nn.GRUCell(in_size + context_size, hidden_size)
    self.radius = radius
    self.hidden_size = hidden_size
    self.context_size = context_size
    self.attention = MILAttention(
      in_size + hidden_size + distance_size,
      context_size, attention_size, heads
    )
    self.angle_lookup = AngleProject(hidden_size, 3)
    self.position_lookup = LocalPositionLookup()

  def collect_context(self, data, results, angles, positions, idx):
    context_features = []
    context_indices = []
    for idy in range(results.size(0)):
      subpositions = positions[idy].detach().numpy()
      last_position = subpositions[idx - 1, :, :]
      tree = cKDTree(subpositions[:idx, 1, :])
      neighbours = tree.query_ball_point(last_position[1, :], self.radius)
      neighbour_states = results[idy, neighbours, :]
      neighbour_inputs = data[idy, neighbours, :]
      neighbour_angles = angles[idy, neighbours, :]
      neighbour_positions = positions[idy, neighbours, :, :]
      last_position = positions[idy, idx - 1]
      neighbour_distances = (last_position - neighbour_positions).norm(dim=1)
      neighbour_features = torch.cat((
        neighbour_states,
        neighbour_inputs,
        neighbour_angles,
        neighbour_distances
      ), dim=1)
      neighbour_indices = idy * torch.ones(
        len(neighbours), dtype=torch.long, device=results.device
      )
      context_features.append(neighbour_features)
      context_indices.append(neighbour_indices)
    context_features = torch.cat(context_features, dim=0)
    context_indices = torch.cat(context_indices, dim=0)
    return context_features, context_indices

  def forward(self, inputs, structure):
    data, _, index, _ = scatter.pad(inputs, structure)

    results = torch.zeros(data.size(0), data.size(1), self.hidden_size, dtype=inputs.dtype, device=inputs.device)#[]
    angles = torch.zeros(data.size(0), data.size(1), 3, dtype=inputs.dtype, device=inputs.device)#[]
    positions = torch.zeros(data.size(0), data.size(1), 3, 3, dtype=inputs.dtype, device=inputs.device)#[]
    state = torch.zeros(data.size(0), self.hidden_size, dtype=inputs.dtype, device=inputs.device)
    context = torch.zeros(data.size(0), self.context_size, dtype=inputs.dtype, device=inputs.device)

    for idx in range(data.size(1)):
      current_input = data[:, idx, :]
      current_input = torch.cat((current_input, context), dim=1)
      state = self.cell(current_input, state)
      angle = self.angle_lookup(state)
      tertiary = self.position_lookup(angle, positions, idx)

      # update lists
      results[:, idx, :] = state #.append(state)
      angles[:, idx, :] = angle #.append(angle)
      positions[:, idx, :, :] = tertiary #.append(tertiary)

      # compute new context
      if idx != 0:
        context_features, context_indices = self.collect_context(data, results, angles, positions, idx)
        context = self.attention(context_features, context_indices)

    #pp, aa = positions.clone(), angles.clone()
    positions = scatter.unpad(positions, index)
    angles = scatter.unpad(angles, index)

    return positions, angles#, pp, aa
