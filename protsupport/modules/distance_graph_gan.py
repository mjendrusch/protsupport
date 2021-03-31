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
from protsupport.modules.anglespace import PositionLookup, AngleProject, AngleLookup, AngleSample

from torchsupport.utils.memory import memory_used

class DistanceBlock(nn.Module):
  def __init__(self, size, distance_size, attention_size=128, heads=128,
               hidden_size=128, mlp_depth=3, activation=func.relu_,
               batch_norm=False, dropout=0.1, pre_norm=True,
               normalization=lambda x: x, connected=attention_connected):
    super(DistanceBlock, self).__init__()
    self.batch_norm = batch_norm
    self.attention = ts.NeighbourDotMultiHeadAttention(size + 1, distance_size, 128, heads=8, query_size=distance_size, normalization=normalization)
    self.local = MLP(
      distance_size, distance_size,
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

  def forward(self, features, distance_features, structure):
    normed = self.bn(distance_features)
    out = distance_features + self.dropout(self.attention(features, normed, structure))
    out = out + self.dropout(self.local(self.local_bn(out)))
    return out

class ReverseDistanceBlock(nn.Module):
  def __init__(self, size, distance_size, attention_size=128, heads=128,
               hidden_size=128, mlp_depth=3, activation=func.relu_,
               batch_norm=False, dropout=0.1, pre_norm=True,
               normalization=lambda x: x, connected=attention_connected):
    super(ReverseDistanceBlock, self).__init__()
    self.batch_norm = batch_norm
    self.attention = ts.NeighbourDotMultiHeadAttention(
      distance_size + 2, size, attention_size, query_size=size, heads=heads,
      normalization=normalization
    )
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

  def forward(self, features, distance_features, structure):
    normed = self.bn(features)
    out = features + self.dropout(self.attention(distance_features, normed, structure))
    out = out + self.dropout(self.local(self.local_bn(out)))
    return out

class DistanceTransformerEncoderBlock(nn.Module):
  def __init__(self, size, distance_size, attention_size=128, heads=128,
               hidden_size=128, mlp_depth=3, activation=func.relu_,
               batch_norm=False, dropout=0.1, pre_norm=True,
               normalization=lambda x: x, connected=attention_connected):
    super(DistanceTransformerEncoderBlock, self).__init__()
    self.fw = DistanceBlock(
      size, distance_size,
      attention_size=attention_size, heads=heads,
      hidden_size=hidden_size, mlp_depth=mlp_depth,
      activation=activation, batch_norm=batch_norm,
      dropout=dropout, pre_norm=pre_norm,
      normalization=normalization, connected=connected
    )
    self.rv = ReverseDistanceBlock(
      size, distance_size,
      attention_size=attention_size, heads=heads,
      hidden_size=hidden_size, mlp_depth=mlp_depth,
      activation=activation, batch_norm=batch_norm,
      dropout=dropout, pre_norm=pre_norm,
      normalization=normalization, connected=connected
    )
    self.conv = LocalFeatures(size, size, depth=3)

  def forward(self, node_features, distance_features, node_structure, distance_structure, subgraph):
    node_pos = irange(subgraph.indices).float() / 64
    node_pos_features = torch.cat((node_features, node_pos[:, None]), dim=1)

    dist_pos_0 = ts.scatter.pairwise_no_pad(lambda x, y: y, node_pos, subgraph.indices)[0].float() / 64
    dist_pos_1 = ts.scatter.pairwise_no_pad(lambda x, y: x, node_pos, subgraph.indices)[0].float() / 64
    dist_pos_features = torch.cat((distance_features, dist_pos_0[:, None], dist_pos_1[:, None]), dim=1)

    distance_features = self.fw(node_pos_features, distance_features + 0.1 * torch.randn_like(distance_features), distance_structure)
    node_features = self.rv(node_features + 0.1 * torch.randn_like(node_features), dist_pos_features, node_structure)

    #indices = node_structure.indices
    #node_features = node_features + ts.scatter.batched(self.conv, node_features, subgraph.indices)
    return node_features, distance_features

class NeighbourSelector(nn.Module):
  pass

def reverse_structure(structure):
  indices = structure.indices
  connections = structure.connections
  new_indices, ind = connections.sort(dim=0)
  new_connections = indices[ind]
  result = structure.clone()
  result.indices = new_indices
  result.connnections = new_connections
  return result

# def node_structure(subgraph):
#     indices = torch.arange(subgraph.size(0))
#     unique, counts = subgraph.unique(return_counts=True)
#     left = scatter.pairwise_no_pad(lambda x, y: x, indices, subgraph)[0]
#     right = scatter.pairwise_no_pad(lambda x, y: y, indices, subgraph)[0]
#     left_range = torch.arange(left.size(0))
#     right_range = torch.arange(right.size(0))
#     data = torch.cat((left, right), dim=0)
#     node = torch.cat((left_range, right_range), dim=0)
#     val, ind = data.sort(dim=0)
#     offset = torch.repeat_interleave(
#         torch.arange(counts.sum()),
#         torch.repeat_interleave(counts - 1, counts, dim=0), dim=0)
#     to_sort = ind + (ind.max() + 1) * offset
#     v, i = to_sort.sort(dim=0)
#     node = node[ind[i]]
    
#     inv_node, inv_i = node.sort(dim=0)
#     inv_val = val[inv_i]
#     return (val, node), (inv_node, inv_val)

def get_distance_structure(subgraph):
  indices = torch.arange(subgraph.indices.size(0), device=subgraph.indices.device)
  unique, counts = subgraph.indices.unique(return_counts=True)
  left = ts.scatter.pairwise_no_pad(lambda x, y: x, indices, subgraph.indices)[0]
  right = ts.scatter.pairwise_no_pad(lambda x, y: y, indices, subgraph.indices)[0]
  left_range = torch.arange(left.size(0), device=left.device)
  right_range = torch.arange(right.size(0), device=right.device)
  data = torch.cat((left, right), dim=0)
  node = torch.cat((left_range, right_range), dim=0)
  val, ind = data.sort(dim=0)
  offset = torch.repeat_interleave(
    torch.arange(counts.sum(), device=data.device),
    torch.repeat_interleave(counts - 1, counts, dim=0),
    dim=0
  )
  to_sort = ind + (ind.max() + 1) * offset
  v, i = to_sort.sort(dim=0)
  node = node[ind[i]]
  
  inv_node, inv_i = node.sort(dim=0)
  inv_val = val[inv_i]

  node_struc = ts.ScatterStructure(0, 0, val, node)
  dist_struc = ts.ScatterStructure(0, 0, inv_node, inv_val)
  return node_struc, dist_struc

class LocalFeatures(nn.Module):
  def __init__(self, in_size, size, depth=1):
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

class DistanceTransformer(nn.Module):
  def __init__(self, in_size, size, distance_size, attention_size=128,
               heads=128, hidden_size=128, depth=3, mlp_depth=3, dropout=0.1,
               activation=func.relu_, batch_norm=False, pre_norm=True,
               normalization=lambda x: x, connected=attention_connected):
    super(DistanceTransformer, self).__init__()
    self.preprocessor = nn.Linear(in_size, size)
    self.blocks = nn.ModuleList([
      DistanceTransformerEncoderBlock(
        size, distance_size,
        attention_size=attention_size, heads=heads, hidden_size=hidden_size,
        mlp_depth=mlp_depth, activation=activation, batch_norm=batch_norm,
        pre_norm=pre_norm, dropout=dropout, normalization=normalization,
        connected=connected
      )
      for _ in range(depth)
    ])

  def forward(self, features, distance_features, node_structure, dist_structure, subgraph):
    feat_out = self.preprocessor(features)
    dist_out = distance_features
    for block in self.blocks:
      feat_out, dist_out = block(feat_out, dist_out, node_structure, dist_structure, subgraph)
    return feat_out, dist_out

def zroll(inputs):
    result = inputs.roll(1, dims=0)
    result[0] = 0
    return result

def total_until(counts):
    shift = zroll(counts)
    shift = shift.cumsum(dim=0)
    return shift

def irange(indices, reverse=False):
    unique, counts = indices.unique(return_counts=True)
    shift = total_until(counts)
    shift = torch.repeat_interleave(shift, counts)
    result = torch.arange(indices.size(0), device=indices.device) - shift
    if reverse:
        result = torch.repeat_interleave(counts, counts, dim=0) - result - 1
    return result

class DistanceGenerator(nn.Module):
  def __init__(self, in_size, distance_size=128, hidden_size=128, angles=512,
               fragment_size=5, attention_size=128, heads=8, depth=3,
               mlp_depth=3, dropout=0.1, activation=func.relu_, batch_norm=True,
               pre_norm=True, normalization=lambda x: x, connected=attention_connected):
    super(DistanceGenerator, self).__init__()
    self.in_size = in_size
    self.distance_size = 128
    self.angle_lookup = AngleProject(hidden_size, 3)
    self.distance_lookup = nn.Linear(distance_size, 1)
    self.preproc = LocalFeatures(in_size + 1, hidden_size)
    self.transformer = DistanceTransformer(
      in_size, hidden_size, distance_size,
      hidden_size=hidden_size, attention_size=attention_size,
      heads=8, depth=depth, mlp_depth=mlp_depth,
      dropout=dropout, activation=activation,
      batch_norm=batch_norm, pre_norm=pre_norm,
      normalization=normalization, connected=connected
    )
    self.position_lookup = PositionLookup(fragment_size=fragment_size)

  def sample(self, batch_size):
    latents = torch.randn(batch_size, self.in_size)
    distances = torch.randn(batch_size, self.distance_size)
    lengths = torch.randint(32, 64, (batch_size,))
    latents = torch.repeat_interleave(latents, lengths, dim=0)
    #latents += 0.1 * torch.randn_like(latents)
    distance_lengths = lengths * (lengths - 1) // 2
    distances = torch.repeat_interleave(distances, distance_lengths, dim=0)
    indices = torch.arange(0, batch_size, dtype=torch.long)
    indices = torch.repeat_interleave(indices, lengths, dim=0)
    structure = ts.SubgraphStructure(indices)
    structure.lengths = list(lengths)
    return (
      ts.PackedTensor(latents, lengths=list(lengths)),
      ts.PackedTensor(distances, lengths=list(distance_lengths)),
      structure
    )

  def forward(self, sample):
    latent, distances, structure = sample

    position = irange(structure.indices).float() / 64
    position = position[:, None]
    latent = torch.cat((latent, position), dim=1)

    node_struc, dist_struc = get_distance_structure(structure)

    node_out = ts.scatter.batched(self.preproc, latent, structure.indices)
    dist_out = distances

    node_out, dist_out = self.transformer(node_out, dist_out, node_struc, dist_struc, structure)

    angles = self.angle_lookup(node_out)
    #distances, _ = ts.scatter.pairwise_no_pad(lambda x, y: (x - y) ** 2, node_out, structure.indices)
    #distances = (distances * dist_out.sigmoid()).sum(dim=1, keepdim=True)
    distances = self.distance_lookup(dist_out)
    distances = func.softplus(distances)

    return (
      ts.PackedTensor(angles, lengths=list(structure.counts)),
      ts.PackedTensor(distances, lengths=list(structure.counts * (structure.counts - 1) // 2)),
      structure
    )

class DistanceDiscriminator(nn.Module):
  def __init__(self, distance_size=128, hidden_size=128, angles=512,
               fragment_size=5, attention_size=128, heads=8, depth=3,
               mlp_depth=3, dropout=0.1, activation=func.relu_, batch_norm=True,
               pre_norm=True, normalization=lambda x: x, connected=attention_connected):
    super(DistanceDiscriminator, self).__init__()
    self.angle_lookup = nn.Linear(3, hidden_size)
    self.rbf = (0, 20, 64)
    self.distance_lookup = MLP(64, hidden_size, hidden_size=hidden_size, depth=3, batch_norm=False)

    self.angle_result = nn.Linear(hidden_size, 1)
    self.distance_result = nn.Linear(hidden_size, 1)

    self.preproc = LocalFeatures(hidden_size, hidden_size)
    self.transformer = DistanceTransformer(
      hidden_size, hidden_size, hidden_size,
      hidden_size=hidden_size, attention_size=attention_size,
      heads=8, depth=depth, mlp_depth=mlp_depth,
      dropout=dropout, activation=activation,
      batch_norm=batch_norm, pre_norm=pre_norm,
      normalization=normalization, connected=connected
    )

  def forward(self, inputs):
    angles, distances, structure = inputs

    print(distances.mean(), distances.min(), distances.max())
    print(distances.shape, angles.shape)

    node_struc, dist_struc = get_distance_structure(structure)

    node_out = self.angle_lookup(angles)
    dist_out = self.distance_lookup(gaussian_rbf(distances, *self.rbf))

    node_out = ts.scatter.batched(self.preproc, node_out, structure.indices)

    node_out, dist_out = self.transformer(node_out, dist_out, node_struc, dist_struc, structure)

    indices = torch.repeat_interleave(structure.unique, structure.counts * (structure.counts - 1) // 2, dim=0)

    angles = self.angle_result(ts.scatter.mean(node_out, structure.indices))
    distances = self.distance_result(dist_out)

    result = torch.cat((angles, distances), dim=0)

    return result

# class StructuredDiscriminator(nn.Module):
#   def __init__(self, in_size, size, distance_size, sequence_size=20,
#                attention_size=128, heads=128, hidden_size=128, mlp_depth=3,
#                depth=3, max_distance=20, distance_kernels=16, neighbours=15,
#                activation=func.relu_, batch_norm=True, conditional=False,
#                angles=False, dropout=0.1, connected=attention_connected,
#                normalization=spectral_norm):
#     super().__init__()
#     distance_size = distance_size + distance_kernels - 1
#     self.encoder = StructuredTransformerEncoder(
#       size, size, distance_size,
#       attention_size=attention_size, heads=heads, hidden_size=hidden_size,
#       depth=depth, mlp_depth=mlp_depth, activation=activation,
#       batch_norm=batch_norm, dropout=dropout, normalization=normalization,
#       connected=connected
#     )
#     self.angles = angles
#     self.lookup = PositionLookup(fragment_size=10)
#     self.conditional = conditional
#     self.neighbours = neighbours
#     self.activation = activation
#     self.rbf = (0, max_distance, distance_kernels)
#     self.preprocess = LocalFeatures(6, size)
#     self.postprocess = LocalFeatures(size, size)
#     self.result = nn.Linear(2 * size, 1)
#     self.angle_result = MLP(6, 1, hidden_size=32, depth=3, batch_norm=False, normalization=spectral_norm)

#   def orientations(self, tertiary):
#     ors = orientation(tertiary[:, 1].permute(1, 0)).permute(2, 0, 1).contiguous()
#     return ors.view(tertiary.size(0), -1)

#   def knn_structure(self, tertiary, structure):
#     indices = structure.indices
#     unique, count = indices.unique(return_counts=True)
#     pos = tertiary[:, 1]
#     all_neighbours = []
#     all_values = []
#     for index in unique:
#       current = pos[structure.indices == index]
#       closeness = -(current[:, None] - current[None, :]).norm(dim=-1)
#       values, neighbours = closeness.topk(k=self.neighbours, dim=1)
#       all_neighbours.append(neighbours)
#       all_values.append(values)
#     all_neighbours = torch.cat(all_neighbours, dim=0).to(tertiary.device)
#     all_values = torch.cat(all_values, dim=0)
#     return all_values, ts.ConstantStructure(0, 0, all_neighbours)

#   def forward(self, inputs):
#     angles, distances, subgraph = inputs
#     asin = angles.sin()
#     acos = angles.cos()
#     afeat = torch.cat((asin, acos), dim=1)
#     angle_result = self.angle_result(afeat)
#     features = ts.scatter.batched(self.preprocess, afeat, subgraph.indices)
#     tertiary, _ = self.lookup(tertiary, torch.zeros_like(subgraph.indices))
#     ors = self.orientations(tertiary)
#     pos = tertiary[:, 1]
#     inds = torch.arange(0, pos.size(0), dtype=torch.float, device=pos.device).view(-1, 1)
#     distances = torch.cat((pos, ors, inds), dim=1)

#     dist, structure = self.knn_structure(tertiary, subgraph)
#     neighbour_pos = (pos[:, None] - pos[structure.connections] + 1e-6)
#     dist = (neighbour_pos).contiguous()
#     dist = dist.norm(dim=2, keepdim=True)
#     dist = gaussian_rbf(dist, *self.rbf)

#     distance_data = RelativeStructure(structure, self.rbf)
#     relative_data = distance_data.message(
#       distances, distances
#     )
#     relative_structure = OrientationStructure(structure, relative_data)

#     encoding = self.encoder(features, relative_structure)
#     encoding = ts.scatter.batched(self.postprocess, encoding, subgraph.indices)
#     encoding = torch.cat((features, encoding), dim=1)
#     result = self.result(encoding)#self.result(ts.scatter.mean(encoding, subgraph.indices))

# #    result = torch.cat((result, angle_result), dim=0)

#     return result
