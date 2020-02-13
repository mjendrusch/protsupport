import torch
import torch.nn as nn
import torch.nn.functional as func
from torch.nn.utils.spectral_norm import spectral_norm

from torchsupport.modules.basic import MLP, one_hot_encode
import torchsupport.structured as ts

from protsupport.utils.geometry import relative_orientation
from protsupport.modules.rbf import gaussian_rbf
from protsupport.modules.structures import (
  OrientationStructure, MaskedStructure, RelativeStructure
)
from protsupport.modules.transformer import StructuredTransformerEncoder, StructuredTransformerDecoder
from protsupport.modules.transformer import linear_connected, attention_connected
from protsupport.utils.geometry import orientation
from protsupport.modules.anglespace import PositionLookup

from torchsupport.utils.memory import memory_used

class LocalFeatures(nn.Module):
  def __init__(self, in_size, size):
    super().__init__()
    self.preprocess = spectral_norm(nn.Conv1d(in_size, size, 3, padding=1))
    self.blocks = nn.ModuleList([
      spectral_norm(nn.Conv1d(size, size, 3, dilation=idx + 2, padding=idx + 2))
      for idx in range(4)
    ])

  def forward(self, inputs):
    out = func.elu(self.preprocess(inputs))
    for block in self.blocks:
      out = func.elu(block(out))
    return out

class StructuredEnergy(nn.Module):
  def __init__(self, in_size, size, distance_size, sequence_size=20,
               attention_size=128, heads=128, hidden_size=128, mlp_depth=3,
               depth=3, max_distance=20, distance_kernels=16, neighbours=15,
               activation=func.relu_, batch_norm=False, conditional=False,
               angles=False, dropout=0.1, connected=attention_connected,
               normalization=lambda x: x):
    super().__init__()
    distance_size = distance_size + distance_kernels - 1
    self.encoder = StructuredTransformerEncoder(
      size, size, distance_size,
      attention_size=attention_size, heads=heads, hidden_size=hidden_size,
      depth=depth, mlp_depth=mlp_depth, activation=activation,
      batch_norm=batch_norm, normalization=spectral_norm, dropout=0.1,
      connected=connected
    )
    self.angles = angles
    self.lookup = PositionLookup()
    self.conditional = conditional
    self.neighbours = neighbours
    self.activation = activation
    self.rbf = (0, max_distance, distance_kernels)
    self.local_features = LocalFeatures(6, size)
    self.reweighting = spectral_norm(nn.Linear(size, 1))
    self.energy = spectral_norm(nn.Linear(size, 1))

  def orientations(self, tertiary):
    ors = orientation(tertiary[:, 1].permute(1, 0)).permute(2, 0, 1)
    return ors.view(tertiary.size(0), -1)

  def knn_structure(self, tertiary, structure):
    indices = structure.indices
    unique, count = indices.unique(return_counts=True)
    pos = tertiary[:, 1]
    all_neighbours = []
    for index in unique:
      current = pos[structure.indices == index]
      closeness = -(current[:, None] - current[None, :]).norm(dim=-1)
      closeness = closeness + 3 * torch.randn_like(closeness)
      neighbours = closeness.topk(k=self.neighbours, dim=1).indices
      all_neighbours.append(neighbours)
    all_neighbours = torch.cat(all_neighbours, dim=0).to(tertiary.device)
    return ts.ConstantStructure(0, 0, all_neighbours)

  def forward(self, tertiary, sequence, subgraph):
    features = torch.ones(tertiary.size(0), 27, dtype=tertiary.dtype, device=tertiary.device)
    if self.angles:
      angles = tertiary
      asin = angles.sin()
      acos = angles.cos()
      afeat = torch.cat((asin, acos), dim=1)
      features = ts.scatter.batched(self.local_features, afeat, subgraph.indices)
      tertiary, _ = self.lookup(tertiary, torch.zeros_like(subgraph.indices))
    if self.conditional:
      features = sequence
    ors = self.orientations(tertiary)
    pos = tertiary[:, 1]
    inds = torch.arange(0, pos.size(0), dtype=torch.float, device=pos.device).view(-1, 1)
    distances = torch.cat((pos, ors, inds), dim=1)

    structure = self.knn_structure(tertiary, subgraph)

    distance_data = RelativeStructure(structure, self.rbf)
    relative_data = distance_data.message(
      distances, distances
    )
    relative_structure = OrientationStructure(structure, relative_data)

    encoding = self.encoder(features, relative_structure)
    weight = self.reweighting(encoding)
    weight = ts.scatter.softmax(weight, subgraph.indices)
    encoding = ts.scatter.add(weight * encoding, subgraph.indices)
    result = self.energy(encoding)

    return result
