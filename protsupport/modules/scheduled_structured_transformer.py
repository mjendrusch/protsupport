import torch
import torch.nn as nn
import torch.nn.functional as func

from torchsupport.modules.basic import MLP, one_hot_encode
import torchsupport.structured as ts
from torchsupport.modules.gradient import hard_one_hot

from protsupport.utils.geometry import relative_orientation
from protsupport.modules.rbf import gaussian_rbf
from protsupport.modules.structures import (
  OrientationStructure, MaskedStructure, RelativeStructure
)
from protsupport.modules.transformer import StructuredTransformerEncoder, StructuredTransformerDecoder

from torchsupport.utils.memory import memory_used

class ScheduledStructuredTransformer(nn.Module):
  def __init__(self, in_size, out_size, size, distance_size, sequence_size,
               attention_size=128, heads=128, hidden_size=128, mlp_depth=3,
               depth=3, decoder_depth=1, max_distance=20, distance_kernels=16,
               activation=func.relu_, schedule=10, batch_norm=False):
    super(ScheduledStructuredTransformer, self).__init__()
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
      depth=decoder_depth, mlp_depth=mlp_depth, activation=activation,
      batch_norm=batch_norm
    )
    self.sequence_embedding = nn.Linear(out_size, sequence_size)
    self.rbf = (0, max_distance, distance_kernels)
    self.schedule = schedule

  def prepare_sequence(self, sequence):
    sequence = one_hot_encode(
      sequence, list(range(20))
    ).transpose(0, 1).to(sequence.device)
    return self.sequence_embedding(sequence)

  def forward(self, features, sequence, distances, structure):
    distance_data = RelativeStructure(structure, self.rbf)
    relative_data = distance_data.message(
      distances, distances
    )
    relative_structure = OrientationStructure(structure, relative_data)
    encoding = self.encoder(features, relative_structure)

    # initial evaluation
    sequence = self.prepare_sequence(sequence)
    masked_structure = MaskedStructure(
      structure, relative_data, sequence, encoding
    )
    result = self.decoder(encoding, masked_structure)

    # differentiable scheduled sampling
    for idx in range(self.schedule):
      samples = hard_one_hot(result)
      mask = torch.rand(samples.size(0)) < 0.25
      sequence = mask.float() * samples + (1 - mask.float()) * sequence
      masked_structure = MaskedStructure(
        structure, relative_data, sequence, encoding
      )
      result = self.decoder(encoding, masked_structure)

    return result
