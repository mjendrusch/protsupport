import torch
import torch.nn as nn
import torch.nn.functional as func

import torchsupport.structured as ts

from protsupport.utils.geometry import relative_orientation
from protsupport.modules.rbf import gaussian_rbf

class OrientationStructure(ts.ConstantStructure):
  def __init__(self, structure, distances):
    super().__init__(structure.source, structure.target, structure.connections)
    self.distances = distances

  def message(self, source, target):
    return torch.cat((source[self.connections], self.distances), dim=2)

class SequenceOrientationStructure(OrientationStructure):
  def __init__(self, structure, distances, sequence):
    super().__init__(structure, distances)
    self.sequence = sequence

  def message(self, source, target):
    return torch.cat((source[self.connections], self.sequence[self.connections], self.distances), dim=2)

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

class DistanceRelativeStructure(RelativeStructure):
  def compare(self, source, target):
    x, y = source[:, :3], target[:, :3]
    x_i, y_i = target[:, -1], source[:, -1]
    distance = (x - y).norm(dim=1, keepdim=True)
    distance_sin = torch.sin((x_i - y_i) / 10)[:, None]
    distance_cos = torch.cos((x_i - y_i) / 10)[:, None]
    return torch.cat((
      gaussian_rbf(distance, *self.rbf),
      distance_sin, distance_cos
    ), dim=1)

class MaskedStructure(OrientationStructure):
  def __init__(self, structure, distances,
               sequence, encoder_features):
    super().__init__(structure, distances)
    self.encoder_data = encoder_features[self.connections]
    self.index = torch.tensor(list(range(self.connections.size(0))), dtype=torch.long)
    self.index = self.index.view(-1, 1).to(self.encoder_data.device)
    self.pre = (self.connections < self.index).unsqueeze(-1)
    self.post = ~self.pre
    self.encoder_data[self.pre.expand_as(self.encoder_data)] = 0
    self.sequence = sequence[self.connections]
    self.sequence[self.post.expand_as(self.sequence)] = 0

  def message(self, source, target):
    decoder_data = source[self.connections]
    decoder_data[self.post.expand_as(decoder_data)] = 0
    data = self.encoder_data + decoder_data
    data = torch.cat((data, self.distances, self.sequence), dim=2)
    return data

class FlexibleOrientationStructure(ts.ScatterStructure):
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

class FlexibleRelativeStructure(ts.PairwiseData):
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

class FlexibleMaskedStructure(FlexibleOrientationStructure):
  def __init__(self, structure, distances,
               sequence, encoder_features):
    super().__init__(structure, distances)
    self.encoder_data = encoder_features[self.connections]
    self.index = torch.tensor(list(range(self.connections.size(0))), dtype=torch.long)
    self.index = self.index.view(-1).to(self.encoder_data.device)
    self.pre = (self.connections < self.index).unsqueeze(-1)
    self.post = ~self.pre
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
