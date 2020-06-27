import torch
import torch.nn as nn
import torch.nn.functional as func

from torchsupport.modules.basic import MLP, one_hot_encode
import torchsupport.structured as ts

from protsupport.utils.geometry import relative_orientation
from protsupport.modules.rbf import gaussian_rbf
from protsupport.modules.structures import (
  SequenceOrientationStructure, OrientationStructure,
  MaskedStructure, DistanceRelativeStructure, RelativeStructure
)
from protsupport.modules.transformer import StructuredTransformerEncoder, StructuredTransformerDecoder
from protsupport.modules.transformer import attention_connected, linear_connected, assignment_connected

from torchsupport.utils.memory import memory_used

class LocalFeatures(nn.Module):
  def __init__(self, in_size, size):
    super().__init__()
    self.preprocess = nn.Conv1d(in_size, size, 3, padding=1)
    self.blocks = nn.ModuleList([
      nn.Conv1d(size, size, 3, dilation=2 ** (idx % 3), padding=2 ** (idx % 3))
      for idx in range(4)
    ])
    self.bn = nn.ModuleList([
      nn.InstanceNorm1d(size)
      for idx in range(4)
    ])

  def forward(self, inputs):
    out = func.elu(self.preprocess(inputs))
    for bn, block in zip(self.bn, self.blocks):
      out = func.elu(block(bn(out))) + out
    return out

class ConditionalStructuredTransformer(nn.Module):
  def __init__(self, in_size, size, distance_size, sequence_size=20,
               attention_size=128, heads=128, hidden_size=128, mlp_depth=3, sequence=False,
               depth=3, max_distance=20, distance_kernels=16, connected=attention_connected,
               activation=func.relu_, batch_norm=False, local=False, relative=RelativeStructure):
    super().__init__()
    self.sequence = sequence
    distance_size = distance_size + distance_kernels - 1
    if self.sequence:
      distance_size += 21
    self.encoder = StructuredTransformerEncoder(
      in_size + sequence_size + 1, size, distance_size,
      attention_size=attention_size, heads=heads, hidden_size=hidden_size,
      depth=depth, mlp_depth=mlp_depth, activation=activation,
      batch_norm=batch_norm, connected=connected
    )
    self.activation = activation
    self.rbf = (0, max_distance, distance_kernels)
    self.local = None
    if local:
      self.local = LocalFeatures(size + in_size + sequence_size + 1, size)
    self.decoder = nn.Linear(size, sequence_size)
    self.relative = relative

  def forward(self, angle_features, sequence, distances, structure):
    features = torch.cat((angle_features, sequence), dim=1)
    distance_data = self.relative(structure, self.rbf)
    relative_data = distance_data.message(
      distances, distances
    )
    relative_structure = ...
    if self.sequence:
      relative_structure = SequenceOrientationStructure(structure, relative_data, sequence)
    else:
      relative_structure = OrientationStructure(structure, relative_data)
    encoding = self.encoder(features, relative_structure)

    if self.local is not None:
      encoding = self.local(encoding)

    result = self.decoder(encoding)

    return result

class ConditionalPrestructuredTransformer(ConditionalStructuredTransformer):
  def forward(self, angle_features, sequence, pair_features, structure, protein=None):
    features = torch.cat((angle_features, sequence), dim=1)

    # featurize distances
    distance = pair_features[:, :, 0]
    distance = gaussian_rbf(distance.view(-1, 1), *self.rbf).reshape(distance.size(0), distance.size(1), -1)

    print(distance.shape, pair_features.shape)
    pair_features = torch.cat((distance, pair_features[:, :, 1:]), dim=2)

    relative_structure = ...
    if self.sequence:
      relative_structure = SequenceOrientationStructure(structure, pair_features, sequence)
    else:
      relative_structure = OrientationStructure(structure, pair_features)
    encoding = self.encoder(features, relative_structure)
    if self.local is not None and protein is not None:
      combined = torch.cat((encoding, features), dim=1)
      encoding = ts.scatter.batched(self.local, combined, protein.indices)
    result = self.decoder(encoding)

    return result

