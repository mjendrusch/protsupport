import sys
import random

import torch
import torch.nn as nn
import torch.nn.functional as func
from torchsupport.training.training import SupervisedTraining
from torchsupport.modules.basic import one_hot_encode

from protsupport.data.proteinnet import ProteinNetKNN
from protsupport.modules.baseline import Baseline

class SupervisedKNN(ProteinNetKNN):
  def __getitem__(self, index):
    data = super().__getitem__(index)
    structure = data["tertiary"]
    structure = structure.view(-1, structure.size(-1)) / 1000
    angles = data["angles"]
    sin = torch.sin(angles)
    cos = torch.cos(angles)
    primary_subset = data["primary"] - 1
    primary_subset[0] = -1
    num_unknown = random.randint(0, len(primary_subset))
    set_unknown = random.sample(range(len(primary_subset)), num_unknown)
    primary_subset[set_unknown] = -1
    primary_onehot = one_hot_encode(primary_subset, list(range(-1, 20)))
    # positional = data["indices"][None].to(torch.float)
    structure_features = torch.cat((sin, cos, structure, primary_onehot), dim=0)

    primary = data["primary"][0] - 1

    return structure_features, primary

class DebugLoss(nn.Module):
  def __init__(self):
    super().__init__()
    self.loss = nn.CrossEntropyLoss()

  def forward(self, inputs, targets):
    return self.loss(inputs, targets)

if __name__ == "__main__":
  data = SupervisedKNN(sys.argv[1], num_neighbours=15)
  valid_data = SupervisedKNN(sys.argv[2], num_neighbours=15)
  net = Baseline(aa_size=20, in_size=39, hidden_size=100, neighbours=15)
  training = SupervisedTraining(
    net, data, valid_data,
    [DebugLoss()],
    batch_size=512,
    max_epochs=50,
    optimizer=lambda x: torch.optim.Adam(x, lr=1e-2),
    device="cuda:0",
    network_name="baseline-autoregressive",
    valid_callback=lambda x, y, z: x
  )
  final_net = training.train()