import torch
from torch import nn
from torch.nn import functional as func

from torchsupport.modules.structured import connected_entities as ce
from torchsupport.modules.structured import entitynn as enn

class LocalMLP(nn.Module):
  def __init__(self, in_size, out_size,
               hidden_size=128, depth=3, 
               activation=func.relu):
    super(LocalMLP, self).__init__()
    self.preprocessor = nn.Linear(in_size, hidden_size)
    self.postprocessor = nn.Linear(hidden_size, out_size)
    self.blocks = nn.ModuleList([
      nn.Linear(hidden_size, hidden_size)
      for idx in range(depth - 2)
    ])

  def forward(self, inputs):
    out = self.activation(self.preprocessor(inputs))
    for block in self.blocks:
      out = self.activation(block(out))
    return self.activation(self.postprocessor(out))

class LocalWeighting(nn.Module):
  def __init__(self, in_size, out_size,
               hidden_size=128, depth=3,
               activation=func.relu):
    super(LocalWeighting, self).__init__()
    self.value = LocalMLP(
      in_size, out_size,
      hidden_size=hidden_size,
      depth=depth,
      activation=activation
    )
    self.weight = LocalMLP(
      in_size, 1,
      hidden_size=hidden_size,
      depth=depth - 1,
      activation=activation
    )

  def forward(self, inputs):
    return self.value(inputs) * self.weight(inputs)

class BaselineInverseFold(nn.Module):
  def __init__(self, in_size, out_size, hidden_size=128, depth=4):
    super(BaselineInverseFold, self).__init__()
    self.weighting = LocalWeighting(in_size, out_size, depth=depth)
    self.interaction = enn.NeighbourDotAttention(out_size)
    self.mlp = LocalMLP(out_size, 20, hidden_size=128, depth=depth)

  def forward(self, inputs, structure):
    out = self.weighting(inputs)
    out = self.mlp(self.interaction(out, out, structure))
    return out

class StackedInverseFold(nn.Module):
  def __init__(self, size, hidden_size=128, depth=3, outer_depth=3):
    super(StackedInverseFold, self).__init__()
    self.weights = nn.ModuleList([
      LocalWeighting(size, size, depth=depth)
      for idx in range(outer_depth)
    ])
    self.interactions = nn.ModuleList([
      enn.NeighbourAttention(size)
      for idx in range(outer_depth)
    ])
    self.mlp = LocalMLP(size, 20, hidden_size=128, depth=depth)

  def forward(self, inputs, structure):
    out = inputs
    for w, i in zip(self.weights, self.interactions):
      wr = w(out)
      out = i(wr, wr, structure) + out
    return self.mlp(out)
