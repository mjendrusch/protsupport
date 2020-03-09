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
from protsupport.modules.transformer import linear_connected, attention_connected
from protsupport.utils.geometry import orientation
from protsupport.modules.anglespace import PositionLookup, AngleProject

from torchsupport.utils.memory import memory_used

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

class LocalFeatures(nn.Module):
  def __init__(self, in_size, size, depth=4):
    super().__init__()
    self.preprocess = nn.Conv1d(in_size, size, 3, padding=1)
    self.blocks = nn.ModuleList([
      nn.Conv1d(size, size, 3, dilation=idx + 2, padding=idx + 2)
      for idx in range(depth)
    ])
    self.bn = nn.ModuleList([
      nn.InstanceNorm1d(size)
      for idx in range(depth)
    ])

  def forward(self, inputs):
    out = func.elu(self.preprocess(inputs))
    for bn, block in zip(self.bn, self.blocks):
      out = func.elu(block(bn(out)) + out)
    return out

class StructuredGenerator(nn.Module):
  def __init__(self, in_size, hidden_size=800, angles=512, fragment_size=5):
    super(StructuredGenerator, self).__init__()
    self.in_size = in_size
    self.angle_lookup = AngleProject(2 * hidden_size, 3)
    self.preproc = nn.GRUCell(in_size, in_size)
    self.rnn = nn.LSTM(in_size, hidden_size, 2, bidirectional=True, batch_first=True)
    self.position_lookup = PositionLookup(fragment_size=fragment_size)

  def sample(self, batch_size):
    latents = torch.randn(batch_size, self.in_size)
    lengths = torch.randint(32, 200, (batch_size,))
    indices = torch.arange(0, batch_size, dtype=torch.long)
    indices = torch.repeat_interleave(indices, lengths, dim=0)
    structure = ts.SubgraphStructure(indices)
    structure.lengths = list(lengths)
    return latents, structure

  def forward(self, sample):
    latent, structure = sample
    indices = structure.indices

    out = ts.scatter.autoregressive(self.preproc, latent, indices)
    out, _ = ts.scatter.sequential(self.rnn, out, indices)
    angles = self.angle_lookup(out)
    #positions, _ = self.position_lookup(angles, torch.zeros_like(indices))
    return ts.PackedTensor(angles, lengths=list(structure.counts)), structure

class StructuredDiscriminator(nn.Module):
  def __init__(self, in_size, size, distance_size, sequence_size=20,
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
      batch_norm=batch_norm, dropout=dropout, normalization=normalization,
      connected=connected
    )
    self.angles = angles
    self.lookup = PositionLookup(fragment_size=10)
    self.conditional = conditional
    self.neighbours = neighbours
    self.activation = activation
    self.rbf = (0, max_distance, distance_kernels)
    self.preprocess = LocalFeatures(6, size)
    self.postprocess = LocalFeatures(size, size)
    self.result = nn.Linear(2 * size, 1)

  def orientations(self, tertiary):
    ors = orientation(tertiary[:, 1].permute(1, 0)).permute(2, 0, 1).contiguous()
    return ors.view(tertiary.size(0), -1)

  def knn_structure(self, tertiary, structure):
    indices = structure.indices
    unique, count = indices.unique(return_counts=True)
    pos = tertiary[:, 1]
    all_neighbours = []
    all_values = []
    for index in unique:
      current = pos[structure.indices == index]
      closeness = -(current[:, None] - current[None, :]).norm(dim=-1)
      values, neighbours = closeness.topk(k=self.neighbours, dim=1)
      all_neighbours.append(neighbours)
      all_values.append(values)
    all_neighbours = torch.cat(all_neighbours, dim=0).to(tertiary.device)
    all_values = torch.cat(all_values, dim=0)
    return all_values, ts.ConstantStructure(0, 0, all_neighbours)

  def forward(self, inputs):
    tertiary, subgraph = inputs
    angles = tertiary
    asin = angles.sin()
    acos = angles.cos()
    afeat = torch.cat((asin, acos), dim=1)
    print(afeat.shape, asin.shape, acos.shape)
    features = ts.scatter.batched(self.preprocess, afeat, subgraph.indices)
    tertiary, _ = self.lookup(tertiary, torch.zeros_like(subgraph.indices))
    ors = self.orientations(tertiary)
    pos = tertiary[:, 1]
    inds = torch.arange(0, pos.size(0), dtype=torch.float, device=pos.device).view(-1, 1)
    distances = torch.cat((pos, ors, inds), dim=1)

    dist, structure = self.knn_structure(tertiary, subgraph)
    neighbour_pos = (pos[:, None] - pos[structure.connections] + 1e-6)
    dist = (neighbour_pos).contiguous()
    dist = dist.norm(dim=2, keepdim=True)
    dist = gaussian_rbf(dist, *self.rbf)

    distance_data = RelativeStructure(structure, self.rbf)
    relative_data = distance_data.message(
      distances, distances
    )
    relative_structure = OrientationStructure(structure, relative_data)

    encoding = self.encoder(features, relative_structure)
    encoding = ts.scatter.batched(self.postprocess, encoding, subgraph.indices)
    encoding = torch.cat((features, encoding), dim=1)
    result = self.result(encoding)

    return result
