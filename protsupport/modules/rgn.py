import torch
import torch.nn as nn
import torch.nn.functional as func

from torchsupport.modules.residual import ResNetBlock1d
from torchsupport.structured import scatter

from protsupport.modules.anglespace import AngleLookup, PositionLookup, DistanceLookup

class RGN(nn.Module):
  def __init__(self):
    super(RGN, self).__init__()
    pass

class _ConvBlockInner(nn.Module):
  def __init__(self, in_size, hidden_size, dilation=1):
    super(_ConvBlockInner, self).__init__()
    self.blocks = nn.ModuleList([
      nn.Conv1d(in_size, hidden_size, 1),
      nn.Conv1d(
        hidden_size, hidden_size, 3,
        dilation=dilation, padding=dilation
      ),
      nn.Conv1d(hidden_size, in_size, 1)
    ])
    self.bn = nn.ModuleList([
      nn.BatchNorm1d(in_size),
      nn.BatchNorm1d(hidden_size),
      nn.BatchNorm1d(hidden_size)
    ])

  def forward(self, inputs):
    out = inputs
    for bn, block in zip(self.bn, self.blocks):
      out = block(bn(func.relu_(out)))
    return inputs + out

class _ConvBlock(nn.Module):
  def __init__(self, in_size, hidden_size, dilations=None):
    super(_ConvBlock, self).__init__()
    if dilations is None:
      dilations = [1, 2, 4, 8]
    self.blocks = nn.ModuleList([
      _ConvBlockInner(in_size, hidden_size, dilation=dilation)
      for dilation in dilations
    ])

  def forward(self, inputs):
    out = inputs
    for block in self.blocks:
      out = block(out)
    return out

class InteractionConv(nn.Module):
  def __init__(self, in_size, state_size, hidden_size, depth=1):
    super(InteractionConv, self).__init__()
    self.blocks = nn.Sequential(*[
      _ConvBlock(in_size + state_size, hidden_size)
      for idx in range(depth)
    ])
    self.angles = nn.Linear(in_size + state_size, state_size)
    self.hidden = nn.Linear(in_size + state_size, state_size)

  def forward(self, inputs, state, indices):
    out = torch.cat((inputs, state), dim=1)
    out = scatter.batched(self.blocks, out, indices)
    angles = self.angles(out)
    state = self.hidden(out)
    return angles, state

class ConvolutionalGN(nn.Module):
  def __init__(self, in_size, hidden_size=128, depth=3,
               fragment_size=5, angles=60):
    super(ConvolutionalGN, self).__init__()
    self.state = nn.Linear(in_size, hidden_size)
    self.angles = nn.Linear(in_size, hidden_size)
    self.angle_unlookup = nn.Linear(6, hidden_size)
    self.angle_lookup = AngleLookup(hidden_size, angles)
    self.position_lookup = PositionLookup(fragment_size=fragment_size)
    self.blocks = nn.ModuleList([
      InteractionConv(hidden_size, hidden_size, hidden_size // 2, depth=1)
      for idx in range(depth)
    ])

  def forward(self, inputs, structure):
    indices = structure.indices
    out = self.angles(inputs)
    state = self.state(inputs)

    angles_out = []
    #distances_out = []

    for idx, block in enumerate(self.blocks):
      out, state = block(out, state, indices)
      angles = self.angle_lookup(out)
      distances, _ = self.position_lookup(angles, indices)

      #angles_out.append(angles)
      #distances_out.append(distances.clone())

      # TODO: do stuff with distances ...

      unlookup = self.angle_unlookup(
        torch.cat((angles.sin(), angles.cos()), dim=1)
      )
      out = out + unlookup
      
      #if idx != len(self.blocks) - 1:
      #  del distances

    print("asize", angles.size())

    return distances, angles.view(-1)
