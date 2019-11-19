import torch
import torch.nn as nn
import torch.nn.functional as func

from torchsupport.modules.basic import MLP, one_hot_encode
from torchsupport.modules.normalization import AdaptiveLayerNorm
import torchsupport.structured as ts

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
    return features + self.dropout(attention)

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
               batch_norm=False, adaptive=None, dropout=0.1):
    super(StructuredTransformerDecoderBlock, self).__init__()
    self.batch_norm = batch_norm
    self.adaptive = adaptive
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
      if self.adaptive is not None:
        self.bn = AdaptiveLayerNorm(size, adaptive)
        self.local_bn = AdaptiveLayerNorm(size, adaptive)
      else:
        self.bn = nn.LayerNorm(size)
        self.local_bn = nn.LayerNorm(size)

  def forward(self, features, structure):
    latent = []
    if self.adaptive:
      features, latent = features
      latent = [latent]
    inputs = self.activation(self.bn(features, *latent))
    local = self.activation(self.local_bn(self.local(inputs), *latent))
    attention = self.dropout(self.attention(local, local, structure))
    return features + attention

class StructuredTransformerDecoder(nn.Module):
  def __init__(self, out_size, size, distance_size, sequence_size,
               attention_size=128, heads=128, hidden_size=128,
               depth=3, mlp_depth=3, activation=func.relu_,
               batch_norm=False, adaptive=None):
    super(StructuredTransformerDecoder, self).__init__()
    self.postprocessor = nn.Linear(size, out_size)
    self.adaptive = adaptive
    self.blocks = nn.ModuleList([
      StructuredTransformerDecoderBlock(
        size, distance_size, sequence_size,
        attention_size=attention_size, heads=heads, hidden_size=hidden_size,
        mlp_depth=mlp_depth, activation=activation, batch_norm=batch_norm,
        adaptive=adaptive
      )
      for _ in range(depth)
    ])

  def forward(self, features, structure):
    latents = None
    inputs = features
    if self.adaptive is not None:
      features, latents = features
      inputs = (features, latents)
    for block in self.blocks:
      out = block(inputs, structure)
      if self.adaptive is not None:
        inputs = (out, latents)
      else:
        inputs = out
    out = self.postprocessor(out)
    return out
