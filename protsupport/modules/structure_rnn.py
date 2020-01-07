import numpy as np

from scipy.spatial.ckdtree import cKDTree

import torch
import torch.nn as nn
import torch.nn.functional as func

from torchsupport.modules.residual import ResNetBlock1d
from torchsupport.structured import scatter, MILAttention, FullyConnectedScatter

from protsupport.modules.anglespace import AngleLookup, PositionLookup, LocalPositionLookup, DistanceLookup, AngleSample, AngleProject

class StackCell(nn.Module):
  def __init__(self, in_size, hidden_size, depth=3):
    super().__init__()
    self.hidden_size = hidden_size
    self.blocks = nn.ModuleList([
      nn.GRUCell(in_size, hidden_size)
    ] + [
      nn.GRUCell(hidden_size, hidden_size)
      for idx in range(depth - 1)
    ])

  def init_state(self, batch_size, dtype, device):
    results = []
    for _ in self.blocks:
      results.append(torch.zeros(batch_size, self.hidden_size, dtype=dtype, device=device))
    return results

  def forward(self, inputs, state):
    out = inputs
    results = []
    for block, state in zip(self.blocks, state):
      out = block(out, state)
      results.append(out)
    return out, results

class StructureRNN(nn.Module):
  def __init__(self, in_size, distance_size=64, context_size=800,
               hidden_size=800, radius=8, attention_size=128, heads=8,
               depth=5):
    super().__init__()
    self.cell = StackCell(in_size + context_size, hidden_size, depth=depth)#nn.GRUCell(in_size + context_size, hidden_size)
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

    last_position = positions[:, idx]
    distances = (positions[:, :idx + 1, 1, :] - last_position[:, None, 1, :]).norm(dim=-1)
    admissible = (distances < self.radius).nonzero()
    context_indices = admissible[:, 0]
    context_accessor = admissible[:, 1]
    context_inputs = data[context_indices, context_accessor]
    context_results = results[context_indices, context_accessor]
    context_angles = angles[context_indices, context_accessor]
    context_positions = positions[context_indices, context_accessor]
    context_distances = context_positions - last_position[context_indices]
    context_distances = context_distances.norm(dim=-1)

    context_features = torch.cat((
      context_results,
      context_inputs,
      context_angles.sin(),
      context_angles.cos(), 
      context_distances / 8
    ), dim=1)

    # for idy in range(results.size(0)):
    #   subpositions = positions[idy].detach().cpu().numpy()
    #   last_position = subpositions[idx, :, :]
    #   tree = cKDTree(subpositions[:idx + 1, 1, :])
    #   neighbours = tree.query_ball_point(last_position[1, :], self.radius)
    #   neighbours = neighbours[:15]
    #   neighbour_states = results[idy, neighbours, :].clone()
    #   neighbour_inputs = data[idy, neighbours, :].clone()
    #   neighbour_angles = angles[idy, neighbours, :].clone()
    #   neighbour_positions = positions[idy, neighbours, :, :].clone()
    #   last_position = positions[idy, idx].clone()
    #   neighbour_distances = (last_position - neighbour_positions).norm(dim=1)
    #   neighbour_features = torch.cat((
    #     neighbour_states,
    #     neighbour_inputs,
    #     neighbour_angles,
    #     neighbour_distances
    #   ), dim=1)
    #   neighbour_indices = idy * torch.ones(
    #     len(neighbours), dtype=torch.long, device=results.device
    #   )
    #   context_features.append(neighbour_features)
    #   context_indices.append(neighbour_indices)
    # context_features = torch.cat(context_features, dim=0)
    # context_indices = torch.cat(context_indices, dim=0)
    return context_features, context_indices

  def forward(self, inputs, structure):
    data, _, index, _ = scatter.pad(inputs, structure.indices)

    results = torch.zeros(data.size(0), data.size(1), self.hidden_size, dtype=inputs.dtype, device=inputs.device)#[]
    angles = torch.zeros(data.size(0), data.size(1), 3, dtype=inputs.dtype, device=inputs.device)#[]
    positions = torch.zeros(data.size(0), data.size(1), 3, 3, dtype=inputs.dtype, device=inputs.device)#[]
    state = self.cell.init_state(data.size(0), inputs.dtype, inputs.device)#torch.zeros(data.size(0), self.hidden_size, dtype=inputs.dtype, device=inputs.device)
    context = torch.zeros(data.size(0), self.context_size, dtype=inputs.dtype, device=inputs.device)

    for idx in range(data.size(1)):
      current_input = data[:, idx, :]
      current_input = torch.cat((current_input, context), dim=1)
      result, state = self.cell(current_input, state)
      angle = self.angle_lookup(result)
      tertiary = self.position_lookup(angle, positions, idx)

      # update lists
      results[:, idx, :] = result #.append(state)
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
