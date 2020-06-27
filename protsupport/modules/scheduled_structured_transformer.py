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
  def __init__(self, in_size, size, distance_size, sequence_size=20, sequence_embedding_size=64,
               attention_size=128, heads=128, hidden_size=128, mlp_depth=3,
               depth=3, decoder_depth=1, max_distance=20, distance_kernels=16,
               activation=func.relu_, schedule=10, batch_norm=False, relative=RelativeStructure):
    super(ScheduledStructuredTransformer, self).__init__()
    distance_size = distance_size + distance_kernels - 1
    self.encoder = StructuredTransformerEncoder(
      in_size + sequence_size + 1, size, distance_size,
      attention_size=attention_size, heads=heads, hidden_size=hidden_size,
      depth=depth, mlp_depth=mlp_depth, activation=activation,
      batch_norm=batch_norm
    )
    self.decoder = StructuredTransformerDecoder(
      sequence_size, size, distance_size, sequence_embedding_size,
      attention_size=attention_size, heads=heads, hidden_size=hidden_size,
      depth=decoder_depth, mlp_depth=mlp_depth, activation=activation,
      batch_norm=batch_norm
    )
    self.sequence_embedding = nn.Linear(sequence_size, sequence_embedding_size)
    self.rbf = (0, max_distance, distance_kernels)
    self.schedule = schedule
    self.relative = relative

  def prepare_sequence(self, sequence):
    sequence = one_hot_encode(
      sequence, list(range(20))
    ).transpose(0, 1).to(sequence.device)
    return self.sequence_embedding(sequence)

  def sample(self, features, distances, structure):
    # encode
    distance_data = self.relative(structure, self.rbf)
    relative_data = distance_data.message(
      distances, distances
    )
    relative_structure = OrientationStructure(structure, relative_data)
    encoding = self.encoder(features, relative_structure)

    # sampling
    sampled = torch.zeros(encoding.size(0), dtype=torch.long, device=features.device)
    sequence = self.prepare_sequence(sampled)
    masked_structure = MaskedStructure(
      structure, relative_data, sequence, encoding
    )
    result = self.decoder(encoding, masked_structure)
    hard = hard_one_hot(result)
    sampled[0] = hard[0].argmax(dim=0)
    for idx in range(1, len(sampled)):
      sequence = self.prepare_sequence(sampled)
      masked_structure = MaskedStructure(
        structure, relative_data, sequence, encoding
      )
      result = self.decoder(encoding, masked_structure)
      hard = hard_one_hot(result)
      sequence = self.sequence_embedding(hard)
      sampled[idx] = hard[idx].argmax(dim=0)
    return sampled

  def forward(self, features, sequence, distances, structure):
    distance_data = self.relative(structure, self.rbf)
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
      samples = self.sequence_embedding(hard_one_hot(result))
      mask = torch.rand(samples.size(0), device=samples.device) < 0.25
      mask = mask.unsqueeze(1).float()
      sequence = mask * samples + (1 - mask) * sequence
      masked_structure = MaskedStructure(
        structure, relative_data, sequence, encoding
      )
      result = self.decoder(encoding, masked_structure)

    return result

class ScheduledPrestructuredTransformer(ScheduledStructuredTransformer):
  def forward(self, features, sequence, pair_features, structure, protein=None):
    # featurize distances
    distance = pair_features[:, :, 0]
    distance = gaussian_rbf(distance.view(-1, 1), *self.rbf).reshape(distance.size(0), distance.size(1), -1)

    pair_features = torch.cat((distance, pair_features[:, :, 1:]), dim=2)

    relative_structure = OrientationStructure(structure, pair_features)
    encoding = self.encoder(features, relative_structure)

    # initial evaluation
    sequence = self.prepare_sequence(sequence)
    masked_structure = MaskedStructure(
      structure, pair_features, sequence, encoding
    )
    result = self.decoder(encoding, masked_structure)

    # differentiable scheduled sampling
    for idx in range(self.schedule):
      samples = self.sequence_embedding(hard_one_hot(result))
      mask = torch.rand(samples.size(0), device=samples.device) < 0.25
      mask = mask.unsqueeze(1).float()
      sequence = mask * samples + (1 - mask) * sequence
      masked_structure = MaskedStructure(
        structure, relative_data, sequence, encoding
      )
      result = self.decoder(encoding, masked_structure)

    return result

