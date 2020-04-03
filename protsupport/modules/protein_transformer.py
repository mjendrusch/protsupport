import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as func
from torch.nn.utils.spectral_norm import spectral_norm

from torchsupport.modules.basic import MLP, one_hot_encode
import torchsupport.structured as ts

from protsupport.utils.geometry import relative_orientation
from protsupport.modules.rbf import gaussian_rbf
from protsupport.modules.structures import (
  FlexibleOrientationStructure, MaskedStructure, FlexibleRelativeStructure
)
from protsupport.modules.transformer import linear_connected, attention_connected
from protsupport.utils.geometry import orientation
from protsupport.modules.anglespace import PositionLookup, AngleProject

from torch.distributions import OneHotCategorical
from torchsupport.distributions import VonMises, Mixture

class StructuredTransformerEncoderBlock(nn.Module):
  def __init__(self, size, distance_size, attention_size=128, heads=128,
               hidden_size=128, mlp_depth=3, activation=func.relu_,
               batch_norm=False, dropout=0.1, pre_norm=True,
               normalization=lambda x: x, connected=attention_connected):
    super(StructuredTransformerEncoderBlock, self).__init__()
    self.pre_norm = pre_norm
    self.batch_norm = batch_norm
    self.attention = connected(size, distance_size, attention_size, heads, normalization)
    self.local = MLP(
      size, size,
      hidden_size=hidden_size,
      depth=mlp_depth,
      activation=activation,
      batch_norm=False,
      normalization=normalization
    )
    self.activation = activation
    self.dropout = lambda x: x
    if dropout is not None:
      self.dropout = nn.Dropout(dropout)
    self.bn = lambda x: x
    self.local_bn = lambda x: x
    if self.batch_norm:
      self.bn = nn.LayerNorm(size)
      self.local_bn = nn.LayerNorm(size)

  def forward(self, features, structure):
    if self.pre_norm:
      normed = self.bn(features)
      out = features + self.dropout(self.attention(normed, normed, structure))
      out = out + self.dropout(self.local(self.local_bn(out)))
    else:
      out = features + self.dropout(self.attention(features, features, structure))
      out = self.bn(out)
      out = out + self.dropout(self.local(out))
      out = self.local_bn(out)
    return out

class StructuredTransformerEncoder(nn.Module):
  def __init__(self, in_size, size, distance_size, attention_size=128,
               heads=128, hidden_size=128, depth=3, mlp_depth=3, dropout=0.1,
               activation=func.relu_, batch_norm=False, pre_norm=True,
               normalization=lambda x: x, connected=attention_connected):
    super(StructuredTransformerEncoder, self).__init__()
    self.preprocessor = nn.Linear(in_size, size)
    self.blocks = nn.ModuleList([
      StructuredTransformerEncoderBlock(
        size, distance_size,
        attention_size=attention_size, heads=heads, hidden_size=hidden_size,
        mlp_depth=mlp_depth, activation=activation, batch_norm=batch_norm,
        pre_norm=pre_norm, dropout=dropout, normalization=normalization,
        connected=connected
      )
      for _ in range(depth)
    ])

  def forward(self, features, structure):
    out = self.preprocessor(features)
    for block in self.blocks:
      out = block(out, structure)
    return out

class ProteinTransformer(nn.Module):
  def __init__(self, in_size, size, distance_size, mix=10, schedule=2, radius=8,
               attention_size=128, heads=128, hidden_size=128, mlp_depth=3,
               depth=3, max_distance=20, distance_kernels=16, neighbours=15,
               activation=func.relu_, batch_norm=True, conditional=False,
               angles=False, dropout=0.1, connected=attention_connected,
               normalization=lambda x: x):
    super().__init__()
    distance_size = distance_size + distance_kernels - 1
    self.encoder = StructuredTransformerEncoder(
      size, size, distance_size,
      attention_size=attention_size, heads=heads, hidden_size=hidden_size,
      depth=depth, mlp_depth=mlp_depth, activation=activation,
      batch_norm=batch_norm, dropout=0.1, normalization=lambda x: x,
      connected=connected
    )
    self.schedule = schedule
    self.mix = mix
    self.radius = radius
    self.angles = angles
    self.lookup = PositionLookup(fragment_size=10)
    self.conditional = conditional
    self.neighbours = neighbours
    self.activation = activation
    self.rbf = (0, max_distance, distance_kernels)
    self.preprocess = nn.Linear(6, size)
    self.mean = AngleProject(size, 3 * self.mix)#nn.Linear(size, 3 * self.mix)
    self.log_concentration = nn.Linear(size, 3 * self.mix)
    self.weights = nn.Linear(size, self.mix)
    self.factor = nn.Linear(size, 3 * self.mix)

  def orientations(self, tertiary):
    ors = orientation(tertiary[:, 1].permute(1, 0)).permute(2, 0, 1).contiguous()
    return ors.view(tertiary.size(0), -1)

  def autoregressive_structure(self, tertiary):
    pos = tertiary[:, 1]
    index_range = torch.arange(pos.size(0))
    distance = (pos[:, None] - pos[None, :]).norm(dim=-1).cpu()
    causal_distance = (index_range[:, None] - index_range[None, :])
    within_ball = torch.roll(distance <= self.radius, 1, dims=0)
    within_ball[0] = False
    causal = causal_distance > 0
    admitted = causal * within_ball
    admitted_indices = admitted.nonzero()
    indices = admitted_indices[:, 0]
    connections = admitted_indices[:, 1]

    return ts.ScatterStructure(0, 0, indices, connections, node_count=pos.size(0))

  def mixture_of_von_mises(self, means, concentrations, weights):
    parameters = [
      (means[:, idx], concentrations[:, idx])
      for idx in range(self.mix)
    ]
    distribution = Mixture([
      VonMises(mean, conc)
      for mean, conc, in parameters
    ], weights)
    return distribution.sample()

  def sample_angles(self, mean, concentration, factor, weights):
    weights = OneHotCategorical(probs=weights).sample()
    factor = factor[torch.arange(factor.size(0)), :, weights.argmax(dim=1)]
    angles_0 = self.mixture_of_von_mises(mean[:, 0], concentration[:, 0], weights)
    mean[:, 1] = mean[:, 1] + (factor[:, 0] * angles_0).unsqueeze(-1)

    angles_1 = self.mixture_of_von_mises(mean[:, 1], concentration[:, 1], weights)
    mean[:, 2] = mean[:, 2] + (factor[:, 1] * angles_0 + factor[:, 2] * angles_1).unsqueeze(-1)

    angles_2 = self.mixture_of_von_mises(mean[:, 2], concentration[:, 2], weights)

    angles = torch.cat((angles_0[:, None], angles_1[:, None], angles_2[:, None]), dim=1)
    return angles

  def single_forward(self, angles, tertiary, structure, subgraph):
    # TODO: add linear dependency between angles following PixelCNN++
    previous_angles = angles.roll(1, dims=0)
    previous_angles[0] = 0
    asin = previous_angles.sin()
    acos = previous_angles.cos()
    afeat = torch.cat((asin, acos), dim=1)
    features = self.preprocess(afeat)
    ors = self.orientations(tertiary)
    pos = tertiary[:, 1]
    inds = torch.arange(0, pos.size(0), dtype=torch.float, device=pos.device).view(-1, 1)
    distances = torch.cat((pos, ors, inds), dim=1)

    distance_data = FlexibleRelativeStructure(structure, self.rbf)
    relative_data, _, _ = distance_data.message(
      distances, distances.roll(1, dims=0)
    )
    relative_structure = FlexibleOrientationStructure(structure, relative_data)

    encoding = self.encoder(features, relative_structure)
    mean = self.mean(encoding).view(encoding.size(0), 3, self.mix)
    factor = self.factor(encoding).view(encoding.size(0), 3, self.mix)
    mean[:, 1] = mean[:, 1] + factor[:, 0] * angles[:, 0].unsqueeze(-1)
    mean[:, 2] = mean[:, 2] + factor[:, 1] * angles[:, 0].unsqueeze(-1) + factor[:, 2] * angles[:, 1].unsqueeze(-1)
    concentration = 0.1 + 1000 * self.log_concentration(encoding).sigmoid().view(encoding.size(0), 3, self.mix)

    weights = self.weights(encoding).softmax(dim=-1)

    return weights, mean, concentration, factor

  def forward(self, angles, tertiary, structure, subgraph):
    weights, mean, concentration, factor = self.single_forward(
      angles, tertiary, structure, subgraph
    )
    angles = angles.clone()
    for idx in range(self.schedule):
      sampled_angles = self.sample_angles(mean, concentration, factor, weights)
      admix = torch.rand(angles.size(0)) < 0.5
      angles[admix] = sampled_angles.roll(-1, dims=0)[admix]
      tertiary, _ = self.lookup(
        angles,
        torch.zeros(
          angles.size(0),
          dtype=torch.long,
          device=angles.device
        )
      )
      structure = self.autoregressive_structure(tertiary)
      weights, mean, concentration, factor = self.single_forward(
        angles, tertiary, structure, subgraph
      )

    parameters = [
      (mean[:, :, idx], concentration[:, :, idx])
      for idx in range(self.mix)
    ]

    return ((weights, parameters),)

class ProteinEval(ProteinTransformer):
  def forward(self, previous_angles, tertiary, structure):
    asin = previous_angles.sin()
    acos = previous_angles.cos()
    afeat = torch.cat((asin, acos), dim=1)
    features = self.preprocess(afeat)
    ors = self.orientations(tertiary)
    pos = tertiary[:, 1]
    inds = torch.arange(0, pos.size(0), dtype=torch.float, device=pos.device).view(-1, 1)
    distances = torch.cat((pos, ors, inds), dim=1)

    distance_data = FlexibleRelativeStructure(structure, self.rbf)
    relative_data, _, _ = distance_data.message(
      distances, distances.roll(1, dims=0)
    )
    relative_structure = FlexibleOrientationStructure(structure, relative_data)

    encoding = self.encoder(features, relative_structure)
    mean = self.mean(encoding).view(encoding.size(0), 3, self.mix)
    weights = (self.weights(encoding)).softmax(dim=-1)
    factor = self.factor(encoding)
    concentration = 0.1 + 1000 * self.log_concentration(encoding).sigmoid().view(encoding.size(0), 3, self.mix)
    #concentration = 10000 * torch.ones_like(concentration)

    # NOTE: Sample the angles autoregressively:
    angles = self.sample_angles(mean, concentration, factor, weights)

    # NOTE: Combine previous angles with newly sampled angles:
    angles = torch.cat((previous_angles, angles[-1].unsqueeze(0)), dim=0)

    # NOTE: rebuild tertiary:
    tertiary, _ = self.lookup(angles, torch.zeros(angles.size(0), dtype=torch.long, device=angles.device))
    structure = self.autoregressive_structure(tertiary)

    return angles, tertiary, structure

  def sample(self, size=32, start=5):
    angles = torch.zeros(start, 3, dtype=torch.float, device="cuda:0")
    tertiary, _ = self.lookup(angles, torch.zeros(angles.size(0), dtype=torch.long, device=angles.device))
    structure = self.autoregressive_structure(tertiary)

    for idx in range(2 * start):
      angles, tertiary, structure = self(angles, tertiary, structure)
    
    angles = angles[2 * start:]
    tertiary, _ = self.lookup(angles, torch.zeros(angles.size(0), dtype=torch.long, device=angles.device))
    structure = self.autoregressive_structure(tertiary)

    # angles, tertiary, structure = self(angles, tertiary, structure)
    # angles = angles[:-1]
    # tertiary, _ = self.lookup(angles, torch.zeros(angles.size(0), dtype=torch.long, device=angles.device))
    # structure = self.autoregressive_structure(tertiary)
    for idx in range(size - start):
      angles, tertiary, structure = self(angles, tertiary, structure)
    return angles, tertiary
