import torch
import torch.nn as nn
import torch.nn.functional as func

from torchsupport.modules.basic import MLP, one_hot_encode
from torchsupport.modules.normalization import AdaptiveLayerNorm
import torchsupport.structured as ts

def attention_connected(size, distance_size, attention_size, heads, normalization):
  return ts.NeighbourDotMultiHeadAttention(
    size + distance_size, size, attention_size, query_size=size, heads=heads,
    normalization=normalization
  )

def linear_connected(size, distance_size, attention_size, heads, normalization):
  return ts.NeighbourLinear(size + distance_size, size, normalization=normalization)

def assignment_connected(size, distance_size, attention_size, heads, normalization):
  return ts.NeighbourAssignment(size + distance_size, size, size, attention_size, normalization=normalization)

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

class StructuredTransformerDecoderBlock(nn.Module):
  def __init__(self, size, distance_size, sequence_size, attention_size=128,
               heads=128, hidden_size=128, mlp_depth=3, activation=func.relu_,
               batch_norm=False, adaptive=None, dropout=0.1, pre_norm=True,
               normalization=lambda x: x):
    super(StructuredTransformerDecoderBlock, self).__init__()
    self.batch_norm = batch_norm
    self.pre_norm = pre_norm
    self.adaptive = adaptive
    self.attention = ts.NeighbourDotMultiHeadAttention(
      size + distance_size + sequence_size, size, attention_size, query_size=size, heads=heads,
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
    if self.pre_norm:
      normed = self.bn(features, *latent)
      out = features + self.dropout(self.attention(normed, normed, structure))
      out = out + self.dropout(self.local(self.local_bn(out, *latent)))
    else:
      out = features + self.dropout(self.attention(features, features, structure))
      out = self.bn(out, *latent)
      out = out + self.dropout(self.local(out))
      out = self.local_bn(out, *latent)
    return out

class StructuredTransformerDecoder(nn.Module):
  def __init__(self, out_size, size, distance_size, sequence_size,
               attention_size=128, heads=128, hidden_size=128,
               depth=3, mlp_depth=3, activation=func.relu_, dropout=0.1,
               batch_norm=False, adaptive=None, pre_norm=True,
               normalization=lambda x: x):
    super(StructuredTransformerDecoder, self).__init__()
    self.postprocessor = nn.Linear(size, out_size)
    self.adaptive = adaptive
    self.blocks = nn.ModuleList([
      StructuredTransformerDecoderBlock(
        size, distance_size, sequence_size,
        attention_size=attention_size, heads=heads, hidden_size=hidden_size,
        mlp_depth=mlp_depth, activation=activation, batch_norm=batch_norm,
        adaptive=adaptive, pre_norm=pre_norm, dropout=dropout,
        normalization=normalization
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
