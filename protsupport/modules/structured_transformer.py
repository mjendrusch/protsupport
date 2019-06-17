import torch
import torch.nn as nn
import torch.nn.functional as func

from torchsupport.modules.basic import MLP
import torchsupport.modules.structured as ts

from protsupport.utils.geometry import relative_orientation

class OrientationStructure(ts.PairwiseStructure):
  # def compare(self, source, target):
  #   position = source[]
  pass

class MaskedStructure(ts.PairwiseStructure):
  def __init__(self, structure, distances, local, sequence, encoder_features):
    pass

class StructuredTransformerEncoderBlock(nn.Module):
  def __init__(self, size, distance_size, attention_size=128, heads=128,
               hidden_size=128, mlp_depth=3, activation=func.relu_,
               batch_norm=False):
    super(StructuredTransformerEncoderBlock, self).__init__()
    self.batch_norm = batch_norm
    self.attention = ts.NeighbourDotMultiHeadAttention(
      size + distance_size, size, attention_size, heads=heads
    )
    self.local = MLP(size, size, hidden_size=hidden_size, depth=mlp_depth)
    self.activation = activation
    self.bn = lambda x: x
    self.local_bn = lambda x: x
    self.attention_bn = lambda x: x
    if self.batch_norm:
      self.bn = nn.BatchNorm1d(size)
      self.local_bn = nn.BatchNorm1d(size)
      self.attention_bn = nn.BatchNorm1d(size)

  def forward(self, features, distances, structure):
    inputs = self.activation(self.bn(features))
    local = self.activation(self.bn(self.local(inputs)))
    edge_structure = OrientationStructure(structure, distances, local)
    attention = self.attention(features, edge_structure)
    return features + attention

class StructuredTransformerEncoder(nn.Module):
  def __init__(self, in_size, size, distance_size, attention_size=128,
               heads=128, hidden_size=128, depth=3, mlp_depth=3,
               activation=func.relu_, batch_norm=False):
    self.preprocessor = nn.Linear(in_size, size)
    self.blocks = nn.ModuleList([
      StructuredTransformerEncoderBlock(
        size, distance_size,
        attention_size=attention_size, heads=heads, hidden_size=hidden_size,
        mlp_depth=mlp_depth, activation=activation, batch_norm=batch_norm
      )
      for _ in range(depth)
    ])

  def forward(self, features, distances, structure):
    out = self.preprocessor(features)
    for block in self.blocks:
      out = block(out, distances, structure)
    return out

class StructuredTransformerDecoderBlock(nn.Module):
  def __init__(self, size, distance_size, sequence_size, attention_size=128,
               heads=128, hidden_size=128, mlp_depth=3, activation=func.relu_,
               batch_norm=False):
    super(StructuredTransformerDecoderBlock, self).__init__()
    self.batch_norm = batch_norm
    self.attention = ts.NeighbourDotMultiHeadAttention(
      size + distance_size + sequence_size, size, attention_size, heads=heads
    )
    self.local = MLP(size, size, hidden_size=hidden_size, depth=mlp_depth)
    self.activation = activation
    self.bn = lambda x: x
    self.local_bn = lambda x: x
    self.attention_bn = lambda x: x
    if self.batch_norm:
      self.bn = nn.BatchNorm1d(size)
      self.local_bn = nn.BatchNorm1d(size)
      self.attention_bn = nn.BatchNorm1d(size)

  def forward(self, features, distances, sequence, encoder_features, structure):
    inputs = self.activation(self.bn(features))
    local = self.activation(self.bn(self.local(inputs)))
    edge_structure = MaskedStructure(
      structure, distances, sequence, encoder_features, local
    )
    attention = self.attention(features, edge_structure)
    return features + attention

class StructuredTransformerDecoder(nn.Module):
  def __init__(self, out_size, size, distance_size, sequence_size,
               attention_size=128, heads=128, hidden_size=128,
               depth=3, mlp_depth=3, activation=func.relu_,
               batch_norm=False):
    self.postprocessor = nn.Linear(size, out_size)
    self.blocks = nn.ModuleList([
      StructuredTransformerDecoderBlock(
        size, distance_size, sequence_size,
        attention_size=attention_size, heads=heads, hidden_size=hidden_size,
        mlp_depth=mlp_depth, activation=activation, batch_norm=batch_norm
      )
      for _ in range(depth)
    ])

  def forward(self, features, sequence, distances, structure):
    for block in self.blocks:
      out = block(out, distances, sequence, features, structure)
    out = self.postprocessor(features)
    return out
