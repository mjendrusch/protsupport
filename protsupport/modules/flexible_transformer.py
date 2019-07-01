import torch
import torch.nn as nn
import torch.nn.functional as func

from torchsupport.modules.basic import MLP, one_hot_encode
import torchsupport.structured as ts

from protsupport.utils.geometry import relative_orientation
from protsupport.modules.rbf import gaussian_rbf

from torchsupport.utils.memory import memory_used

class OrientationStructure(ts.ScatterStructure):
  def __init__(self, structure, distances):
    super().__init__(
      structure.source, structure.target,
      structure.connections, structure.indices,
      node_count=structure.node_count
    )
    self.distances = distances

  def message(self, source, target):
    source = torch.cat((source[self.connections], self.distances), dim=1)
    target = target[self.indices]
    indices = self.indices
    return source, target, indices, self.node_count

class RelativeStructure(ts.PairwiseData):
  def __init__(self, structure, rbf_params):
    ts.PairwiseData.__init__(self, structure)
    self.rbf = rbf_params

  def compare(self, source, target):
    x, y = target[:, :3], source[:, :3]
    x_o, y_o = target[:, 3:-1].reshape(-1, 3, 3), source[:, 3:-1].reshape(-1, 3, 3)
    x_i, y_i = target[:, -1], source[:, -1]

    distance, direction, rotation = relative_orientation(x, y, x_o, y_o)
    distance_sin = torch.sin((x_i - y_i) / 10)[:, None]
    distance_cos = torch.cos((x_i - y_i) / 10)[:, None]
    return torch.cat((
      gaussian_rbf(distance, *self.rbf),
      direction, rotation, distance_sin, distance_cos
    ), dim=1)

class MaskedStructure(OrientationStructure):
  def __init__(self, structure, distances,
               sequence, encoder_features):
    super().__init__(structure, distances)
    self.encoder_data = encoder_features[self.connections]
    self.index = torch.tensor(list(range(self.connections.size(0))), dtype=torch.long)
    self.index = self.index.view(-1).to(self.encoder_data.device)
    self.pre = (self.connections < self.index).unsqueeze(-1)
    self.post = 1 - self.pre
    self.encoder_data[self.pre.expand_as(self.encoder_data)] = 0
    self.sequence = sequence[self.connections]
    self.sequence[self.post.expand_as(self.sequence)] = 0

  def message(self, source, target):
    decoder_data = source[self.connections]
    target_data = target[self.indices]
    decoder_data[self.post.expand_as(decoder_data)] = 0
    data = self.encoder_data + decoder_data
    data = torch.cat((data, self.distances, self.sequence), dim=1)
    return data, target_data, self.indices, self.node_count

class StructuredTransformerEncoderBlock(nn.Module):
  def __init__(self, size, distance_size, attention_size=128, heads=128,
               hidden_size=128, mlp_depth=3, activation=func.relu_,
               batch_norm=False, dropout=0.1):
    super(StructuredTransformerEncoderBlock, self).__init__()
    self.batch_norm = batch_norm
    self.attention = ts.NeighbourDotMultiHeadAttention(
      size + distance_size, size, attention_size, query_size=size, heads=heads
    )
    self.local = MLP(
      size, size,
      hidden_size=hidden_size,
      depth=mlp_depth,
      activation=activation,
      batch_norm=False
    )
    self.activation = activation
    self.dropout = lambda x: x
    if dropout is not None:
      self.dropout = nn.Dropout(dropout, inplace=True)
    self.bn = lambda x: x
    self.local_bn = lambda x: x
    if self.batch_norm:
      self.bn = nn.LayerNorm(size)
      self.local_bn = nn.LayerNorm(size)

  def forward(self, features, structure):
    inputs = self.activation(self.bn(features))
    local = self.activation(self.local_bn(self.local(inputs)))
    attention = self.attention(local, local, structure)
    return features + attention

class StructuredTransformerEncoder(nn.Module):
  def __init__(self, in_size, size, distance_size, attention_size=128,
               heads=128, hidden_size=128, depth=3, mlp_depth=3,
               activation=func.relu_, batch_norm=False):
    super(StructuredTransformerEncoder, self).__init__()
    self.preprocessor = nn.Linear(in_size, size)
    self.blocks = nn.ModuleList([
      StructuredTransformerEncoderBlock(
        size, distance_size,
        attention_size=attention_size, heads=heads, hidden_size=hidden_size,
        mlp_depth=mlp_depth, activation=activation, batch_norm=batch_norm
      )
      for _ in range(depth)
    ])

  def forward(self, features, structure):
    out = self.preprocessor(features)
    for block in self.blocks:
      out = block(out, structure)
    return out

class StructuredTransformerDecoderBlock(nn.Module):
  def __init__(self, size, distance_size, sequence_size, attention_size=128,
               heads=128, hidden_size=128, mlp_depth=3, activation=func.relu_,
               batch_norm=False, dropout=0.1):
    super(StructuredTransformerDecoderBlock, self).__init__()
    self.batch_norm = batch_norm
    self.attention = ts.NeighbourDotMultiHeadAttention(
      size + distance_size + sequence_size, size, attention_size, query_size=size, heads=heads
    )
    self.local = MLP(
      size, size,
      hidden_size=hidden_size,
      depth=mlp_depth,
      activation=activation,
      batch_norm=False
    )
    self.activation = activation
    self.dropout = lambda x: x
    if dropout is not None:
      self.dropout = nn.Dropout(dropout, inplace=True)
    self.bn = lambda x: x
    self.local_bn = lambda x: x
    if self.batch_norm:
      self.bn = nn.LayerNorm(size)
      self.local_bn = nn.LayerNorm(size)

  def forward(self, features, structure):
    inputs = self.activation(self.bn(features))
    local = self.activation(self.local_bn(self.local(inputs)))
    attention = self.dropout(self.attention(local, local, structure))
    return features + attention

class StructuredTransformerDecoder(nn.Module):
  def __init__(self, out_size, size, distance_size, sequence_size,
               attention_size=128, heads=128, hidden_size=128,
               depth=3, mlp_depth=3, activation=func.relu_,
               batch_norm=False):
    super(StructuredTransformerDecoder, self).__init__()
    self.postprocessor = nn.Linear(size, out_size)
    self.blocks = nn.ModuleList([
      StructuredTransformerDecoderBlock(
        size, distance_size, sequence_size,
        attention_size=attention_size, heads=heads, hidden_size=hidden_size,
        mlp_depth=mlp_depth, activation=activation, batch_norm=batch_norm
      )
      for _ in range(depth)
    ])

  def forward(self, features, structure):
    out = features
    for block in self.blocks:
      out = block(out, structure)
    out = self.postprocessor(out)
    return out

class StructuredTransformer(nn.Module):
  def __init__(self, in_size, out_size, size, distance_size, sequence_size,
               attention_size=128, heads=128, hidden_size=128, mlp_depth=3,
               depth=3, max_distance=20, distance_kernels=16,
               activation=func.relu_, batch_norm=False):
    super(StructuredTransformer, self).__init__()
    distance_size = distance_size + distance_kernels - 1
    self.encoder = StructuredTransformerEncoder(
      in_size, size, distance_size,
      attention_size=attention_size, heads=heads, hidden_size=hidden_size,
      depth=depth, mlp_depth=mlp_depth, activation=activation,
      batch_norm=batch_norm
    )
    self.decoder = StructuredTransformerDecoder(
      out_size, size, distance_size, sequence_size,
      attention_size=attention_size, heads=heads, hidden_size=hidden_size,
      depth=depth, mlp_depth=mlp_depth, activation=activation,
      batch_norm=batch_norm
    )
    self.sequence_embedding = nn.Linear(out_size, sequence_size)
    self.rbf = (0, max_distance, distance_kernels)

  def prepare_sequence(self, sequence):
    sequence = one_hot_encode(
      sequence, list(range(20))
    ).transpose(0, 1).to(sequence.device)
    return self.sequence_embedding(sequence)

  def forward(self, features, sequence, distances, structure):
    distance_data = RelativeStructure(structure, self.rbf)
    relative_data, _, _ = distance_data.message(
      distances, distances
    )
    relative_structure = OrientationStructure(structure, relative_data)
    encoding = self.encoder(features, relative_structure)

    sequence = self.prepare_sequence(sequence)
    masked_structure = MaskedStructure(
      structure, relative_data, sequence, encoding
    )
    result = self.decoder(encoding, masked_structure)
    return result
