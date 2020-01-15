import torch
import torch.nn as nn
import torch.nn.functional as func

from torchsupport.modules.basic import MLP, one_hot_encode
import torchsupport.structured as ts

from protsupport.utils.geometry import relative_orientation
from protsupport.modules.rbf import gaussian_rbf
from protsupport.modules.structures import (
  OrientationStructure, MaskedStructure, RelativeStructure
)
from protsupport.modules.transformer import StructuredTransformerEncoder, StructuredTransformerDecoder

from torchsupport.utils.memory import memory_used

class ConditionalStructuredTransformer(nn.Module):
  def __init__(self, in_size, size, distance_size, sequence_size=20,
               attention_size=128, heads=128, hidden_size=128, mlp_depth=3,
               depth=3, max_distance=20, distance_kernels=16,
               activation=func.relu_, batch_norm=False):
    super().__init__()
    distance_size = distance_size + distance_kernels - 1
    self.encoder = StructuredTransformerEncoder(
      in_size + sequence_size + 1, size, distance_size,
      attention_size=attention_size, heads=heads, hidden_size=hidden_size,
      depth=depth, mlp_depth=mlp_depth, activation=activation,
      batch_norm=batch_norm
    )
    self.activation = activation
    self.rbf = (0, max_distance, distance_kernels)
    self.decoder = nn.Linear(size, sequence_size)

  def forward(self, angle_features, sequence, distances, structure):
    features = torch.cat((angle_features, sequence), dim=1)
    distance_data = RelativeStructure(structure, self.rbf)
    relative_data = distance_data.message(
      distances, distances
    )
    relative_structure = OrientationStructure(structure, relative_data)
    encoding = self.encoder(features, relative_structure)
    result = self.decoder(
      # self.activation(
        encoding)
        # )

    return result
