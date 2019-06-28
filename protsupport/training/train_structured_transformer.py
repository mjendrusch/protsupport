import sys
import random

from matplotlib import pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as func
from torchsupport.training.training import SupervisedTraining
from torchsupport.modules.basic import one_hot_encode

from torchsupport.modules.structured import ConstantStructure, SubgraphStructure
from torchsupport.data.collate import BatchFirst

from protsupport.data.proteinnet import ProteinNet, ProteinNetKNN
from protsupport.utils.geometry import orientation, relative_orientation
from protsupport.modules.structured_transformer import StructuredTransformer

def valid_callback(training, data, predictions):
  inputs, labels = data
  confusion = torch.zeros(20, 20)
  for label, prediction in zip(labels, predictions[0][0]):
    pred = prediction.argmax(dim=0)
    confusion[label, pred] += 1
  fig, ax = plt.subplots()
  ax.imshow(confusion / confusion.sum(dim=1, keepdim=True), cmap="Reds")
  training.writer.add_figure("confusion", fig, training.step_id)

class TransformerNet(ProteinNetKNN):
  def __init__(self, path, num_neighbours=20, n_jobs=1, cache=True):
    ProteinNetKNN.__init__(
      self, path,
      num_neighbours=num_neighbours,
      n_jobs=n_jobs, cache=cache
    )
    self.ors = torch.tensor(
      orientation(self.ters[1].numpy() / 1000).transpose(2, 0, 1),
      dtype=torch.float
    )

  def __getitem__(self, index):
    window = slice(self.index[index], self.index[index + 1])
    inds = self.inds[window]
    primary = self.pris[window] - 1
    evolutionary = self.evos[:, window]
    tertiary = self.ters[:, :, window]
    orientation = self.ors[window, :, :].view(
      window.stop - window.start, -1
    )
    distances = self.ters[1, :, window].transpose(0, 1) / 1000
    indices = torch.tensor(
      range(window.start, window.stop),
      dtype=torch.float
    )
    indices = indices.view(-1, 1)

    orientation = torch.cat((distances, orientation, indices), dim=1)
    angles = self.angs[:, window].transpose(0, 1)

    neighbours = ConstantStructure(0, 0, inds - self.index[index])

    sin = torch.sin(angles)
    cos = torch.cos(angles)
    angle_features = torch.cat((sin, cos), dim=1)

    inputs = (
      BatchFirst(angle_features),
      BatchFirst(primary),
      BatchFirst(orientation),
      neighbours
    )

    return inputs, BatchFirst(primary)

  def __len__(self):
    return ProteinNet.__len__(self)

class DebugLoss(nn.Module):
  def __init__(self):
    super().__init__()
    self.loss = nn.CrossEntropyLoss()

  def forward(self, inputs, targets):
    return self.loss(inputs, targets)

if __name__ == "__main__":
  data = TransformerNet(sys.argv[1], num_neighbours=15)
  valid_data = TransformerNet(sys.argv[2], num_neighbours=15)
  net = StructuredTransformer(
    6, 20, 128, 10, 20,
    attention_size=128, heads=8,
    mlp_depth=2, depth=5, batch_norm=True
  )
  training = SupervisedTraining(
    net, data, valid_data,
    [DebugLoss()],
    batch_size=8,
    max_epochs=1000,
    optimizer=lambda x: torch.optim.Adam(x, lr=1e-5),
    device="cuda:0",
    network_name="structured-transformer",
    valid_callback=valid_callback
  )
  final_net = training.train()
