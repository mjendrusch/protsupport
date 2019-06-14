import sys

import torch
import torch.nn as nn
import torch.nn.functional as func
from torchsupport.training.training import SupervisedTraining

from protsupport.data.proteinnet import ProteinNetKNN
from protsupport.modules.baseline import Baseline

class SupervisedKNN(ProteinNetKNN):
  def __getitem__(self, index):
    data = super().__getitem__(index)
    structure = data["tertiary"]
    structure = structure.view(-1, structure.size(-1))
    positional = data["indices"][None].to(torch.float)
    structure_features = torch.cat((structure, positional), dim=0)

    primary = data["primary"][0]

    return structure_features, primary

class DebugLoss(nn.Module):
  def __init__(self):
    super().__init__()
    self.loss = nn.CrossEntropyLoss()

  def forward(self, inputs, targets):
    return self.loss(inputs, targets)

if __name__ == "__main__":
  data = SupervisedKNN(sys.argv[1])
  valid_data = SupervisedKNN(sys.argv[2])
  net = Baseline(aa_size=21, in_size=13, hidden_size=100, neighbours=20)
  training = SupervisedTraining(
    net, data, valid_data,
    [DebugLoss()],
    batch_size=256,
    max_epochs=50,
    optimizer=lambda x: torch.optim.Adam(x, lr=0.1),
    device="cuda:0",
    network_name="baseline",
    valid_callback=lambda x, y, z: x
  )
  final_net = training.train()
