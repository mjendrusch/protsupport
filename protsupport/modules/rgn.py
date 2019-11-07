import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as func

from torchsupport.modules.residual import ResNetBlock1d
from torchsupport.structured import scatter, Transformer, FullyConnectedScatter

from protsupport.modules.anglespace import AngleLookup, PositionLookup, DistanceLookup, AngleSample

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
      nn.GroupNorm(32, in_size),
      nn.GroupNorm(32, hidden_size),
      nn.GroupNorm(32, hidden_size)
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
               fragment_size=5, angles=60, integrate=5):
    super(ConvolutionalGN, self).__init__()
    self.state = nn.Linear(in_size, hidden_size)
    self.angles = nn.Linear(in_size, hidden_size)
    self.angle_unlookup = nn.Linear(6, hidden_size)
    self.angle_lookup = AngleSample(hidden_size, angles)
    self.position_lookup = PositionLookup(fragment_size=fragment_size)
    self.blocks = nn.ModuleList([
      InteractionConv(hidden_size, hidden_size, hidden_size // 2, depth=1)
      for idx in range(depth)
    ])
    self.transformer = Transformer(hidden_size, hidden_size, hidden_size // 2, attention_size=8)
    self.integrate = integrate

  def forward(self, inputs, structure):
    indices = structure.indices
    out = self.angles(inputs)
    state = self.state(inputs)

    struc = FullyConnectedScatter(structure.indices)

    angles_out = []
    #distances_out = []

    for idx, block in enumerate(self.blocks):
      for idy in range(self.integrate):
        out, state = block(out, state, indices)
        angles = self.angle_lookup(out)
        distances, _ = self.position_lookup(angles, indices)

      #angles_out.append(angles)
      #distances_out.append(distances.clone())

      # TODO: do stuff with distances ...
        #pos = distances.detach().cpu()

        unlookup = self.angle_unlookup(
          torch.cat((angles.sin(), angles.cos()), dim=1)
        )
        out = out + unlookup
      if self.transformer is not None:
        out = self.transformer(out, struc)
      #if idx != len(self.blocks) - 1:
      #  del distances

    print("asize", angles.size())

    return distances, angles

class TransformerGN(nn.Module):
  def __init__(self, in_size, hidden_size=128, depth=3, fragment_size=5):
    super(TransformerGN, self).__init__()
    self.preprocess = nn.Linear(in_size, hidden_size)
    self.angle_lookup = AngleSample(hidden_size, 60)#nn.Linear(hidden_size, 3)
    self.position_lookup = PositionLookup(fragment_size=fragment_size)
    self.blocks = nn.ModuleList([
      Transformer(hidden_size, hidden_size, hidden_size // 2, attention_size=8)
      for idx in range(depth)
    ])

  def forward(self, inputs, structure):
    fcs = FullyConnectedScatter(structure.indices)
    out = self.preprocess(inputs)
    for block in self.blocks:
      out = block(out, fcs)
    angles = self.angle_lookup(out)
    #angles = torch.atan2(angles.sin(), angles.cos())
    positions, _ = self.position_lookup(angles, structure.indices)
    return positions, angles

class ResidualGN(nn.Module):
  def __init__(self, in_size, hidden_size=128, depth=3,
               fragment_size=5, angles=60, integrate=5):
    super(ResidualGN, self).__init__()
    self.state = nn.Linear(in_size, hidden_size)
    self.angles = nn.Linear(in_size, hidden_size)
    self.angle_unlookup = nn.Linear(6, hidden_size)
    self.angle_lookup = nn.Linear(hidden_size, 3)#AngleLookup(hidden_size, angles)
    self.position_lookup = PositionLookup(fragment_size=fragment_size)
    self.blocks = nn.ModuleList([
      InteractionConv(hidden_size, hidden_size, hidden_size // 2, depth=1)
      for idx in range(depth)
    ])
    self.integrate = integrate

  def forward(self, inputs, structure):
    indices = structure.indices
    out = self.angles(inputs)
    state = self.state(inputs)

    angles_out = []
    #distances_out = []

    angles = 2 * np.pi * (torch.rand(inputs.size(0), 3, device=inputs.device) - 0.5)
    for idx, block in enumerate(self.blocks):
      for idy in range(self.integrate):
        unl = self.angle_unlookup(torch.cat((angles.sin(), angles.cos()), dim=1))
        out = state + unl
        out, state = block(out, state, indices)
        angles_logits = self.angle_lookup(out)
        angles = angles + angles_logits
        angles = torch.atan2(angles.sin(), angles.cos())
        distances, _ = self.position_lookup(angles, indices)

        angles = angles + 0.1 * torch.randn_like(angles)
      #angles_out.append(angles)
      #distances_out.append(distances.clone())

      # TODO: do stuff with distances ...

      #unlookup = self.angle_unlookup(
      #  torch.cat((angles.sin(), angles.cos()), dim=1)
      #)
      #out = out + unlookup
      
      #if idx != len(self.blocks) - 1:
      #  del distances

    print("asize", angles.size())

    return distances, angles

class RGN(nn.Module):
  def __init__(self, in_size, hidden_size=800, angles=512, fragment_size=5):
    super(RGN, self).__init__()
    self.angle_lookup = AngleLookup(2 * hidden_size, angles)
    self.rnn = nn.LSTM(in_size, hidden_size, 2, bidirectional=True, batch_first=True)
    self.position_lookup = PositionLookup(fragment_size=fragment_size)
    #self.angle_rnn = nn.LSTM(2 * hidden_size + 6, hidden_size, 2, batch_first=True)

  def forward(self, inputs, structure):
    indices = structure.indices
    out, _ = scatter.sequential(self.rnn, inputs, indices)
    angles = self.angle_lookup(out)
    positions, _ = self.position_lookup(angles, indices)
    return positions, angles

