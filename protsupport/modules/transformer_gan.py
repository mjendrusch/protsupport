import torch
import torch.nn as nn
import torch.nn.functional as func

from torchsupport.modules.basic import MLP, one_hot_encode
import torchsupport.structured as ts
from torchsupport.structured import scatter
from torchsupport.modules.gradient import hard_one_hot
from torchsupport.structured import PackedTensor

from protsupport.utils.geometry import relative_orientation
from protsupport.modules.rbf import gaussian_rbf
from protsupport.modules.structures import (
  OrientationStructure, MaskedStructure, RelativeStructure
)
from protsupport.modules.transformer import StructuredTransformerEncoder, StructuredTransformerDecoder

from torchsupport.utils.memory import memory_used

class TransformerGenerator(nn.Module):
  def __init__(self, in_size, out_size, size, distance_size, sequence_size,
               attention_size=128, heads=128, hidden_size=128, mlp_depth=3,
               depth=3, max_distance=20, distance_kernels=16, latent_size=256,
               activation=func.relu_, batch_norm=False):
    super(TransformerGenerator, self).__init__()
    distance_size = distance_size + distance_kernels - 1
    self.latent_size = 256
    self.encoder = StructuredTransformerEncoder(
      in_size, size, distance_size,
      attention_size=attention_size, heads=heads, hidden_size=hidden_size,
      depth=depth, mlp_depth=mlp_depth, activation=activation,
      batch_norm=batch_norm
    )
    self.decoder = StructuredTransformerDecoder(
      out_size, size, distance_size, 0,#sequence_size,
      attention_size=attention_size, heads=heads, hidden_size=hidden_size,
      depth=depth, mlp_depth=mlp_depth, activation=activation,
      batch_norm=batch_norm, adaptive=self.latent_size
    )
    self.rbf = (0, max_distance, distance_kernels)

  def sample(self, batch_size):
    latents = torch.randn(4, self.latent_size)
    return latents

  def forward(self, data):
    latents, (angle_features, sequence, mask, distances, structure) = data
    features = torch.cat((angle_features.tensor, sequence.tensor, mask.tensor.unsqueeze(-1)), dim=1)
    distance_data = RelativeStructure(structure, self.rbf)
    relative_data = distance_data.message(
      distances.tensor, distances.tensor
    )
    relative_structure = OrientationStructure(structure, relative_data)
    encoding = self.encoder(features, relative_structure)
    result = self.decoder((encoding, latents), relative_structure)
    result[mask.tensor > 0] = torch.log(sequence.tensor)[mask.tensor > 0]
    return (angle_features, sequence, mask, distances, structure), PackedTensor(result, box=True, lengths=sequence.lengths)

class TransformerDiscriminator(nn.Module):
  def __init__(self, in_size, out_size, size, distance_size,
               attention_size=128, heads=128, hidden_size=128, mlp_depth=3,
               depth=3, max_distance=20, distance_kernels=16, latent_size=256,
               activation=func.relu_, batch_norm=False):
    super(TransformerDiscriminator, self).__init__()
    distance_size = distance_size + distance_kernels - 1
    self.encoder = StructuredTransformerEncoder(
      in_size, size, distance_size,
      attention_size=attention_size, heads=heads, hidden_size=hidden_size,
      depth=2 * depth, mlp_depth=mlp_depth, activation=activation,
      batch_norm=batch_norm
    )
    self.verdict = nn.Linear(size, 1)
    self.rbf = (0, max_distance, distance_kernels)

  def forward(self, data):
    (angle_features, sequence, mask, distances, structure), sequence_logits = data
    features = torch.cat((angle_features.tensor, sequence.tensor, mask.tensor.unsqueeze(-1)), dim=1)
    if sequence_logits.tensor.is_floating_point():
      sequence = hard_one_hot(sequence_logits.tensor)
    else:
      batch_size = sequence_logits.tensor.size(0)
      sequence = torch.zeros(batch_size, 20)
      sequence[torch.arange(0, batch_size), sequence_logits.tensor.view(-1)] = 1
      sequence = sequence.to(features.device)

    distance_data = RelativeStructure(structure, self.rbf)
    relative_data = distance_data.message(
      distances.tensor, distances.tensor
    )
    relative_structure = OrientationStructure(structure, relative_data)
    features = torch.cat((features, sequence), dim=1)
    result = self.encoder(features, relative_structure)
    verdict = self.verdict(result)
    return verdict
