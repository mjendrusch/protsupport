import sys
import random

from matplotlib import pyplot as plt

from scipy.spatial import cKDTree

import torch
import torch.nn as nn
import torch.nn.functional as func
from torchsupport.training.training import SupervisedTraining

from torchsupport.structured import PackedTensor, ScatterStructure
from torchsupport.structured import DataParallel as SDP

from protsupport.data.proteinnet import ProteinNet, ProteinNetKNN
from protsupport.utils.geometry import orientation
from protsupport.modules.flexible_transformer import FlexibleTransformer

def valid_callback(training, data, predictions):
  inputs, labels = data
  confusion = torch.zeros(20, 20)
  for label, prediction in zip(labels, predictions[0][0]):
    pred = prediction.argmax(dim=0)
    confusion[label, pred] += 1
  fig, ax = plt.subplots()
  ax.imshow(confusion / confusion.sum(dim=1, keepdim=True), cmap="Reds")
  training.writer.add_figure("confusion", fig, training.step_id)
  # for name, parameter in training.net.named_parameters():
  #   training.writer.add_histogram(f"phist {name}", parameter.detach().cpu().numpy(), training.step_id)

class TransformerNet(ProteinNetKNN):
  def __init__(self, path, num_neighbours=20, n_jobs=1, cache=True):
    ProteinNetKNN.__init__(
      self, path,
      num_neighbours=num_neighbours,
      n_jobs=n_jobs, cache=cache
    )
    self.ors = torch.tensor(
      orientation(self.ters[1].numpy() / 100).transpose(2, 0, 1),
      dtype=torch.float
    )

  def __getitem__(self, index):
    window = slice(self.index[index], self.index[index + 1])
    window = window[:500]
    inds = self.inds[window]
    primary = self.pris[window] - 1
    evolutionary = self.evos[:, window]
    tertiary = self.ters[:, :, window]
    orientation = self.ors[window, :, :].view(
      window.stop - window.start, -1
    )
    distances = self.ters[1, :, window].transpose(0, 1) / 100
    indices = torch.tensor(
      range(window.start, window.stop),
      dtype=torch.float
    )
    indices = indices.view(-1, 1)

    orientation = torch.cat((distances, orientation, indices), dim=1)
    angles = self.angs[:, window].transpose(0, 1)

    tree = cKDTree(distances)
    connections = tree.query_ball_tree(tree, r=8.0)

    neighbours = ScatterStructure.from_connections(
      0, 0, connections
    )

    sin = torch.sin(angles)
    cos = torch.cos(angles)
    angle_features = torch.cat((sin, cos), dim=1)

    inputs = (
      PackedTensor(angle_features),
      PackedTensor(primary),
      PackedTensor(orientation),
      neighbours
    )

    return inputs, PackedTensor(primary)

  def __len__(self):
    return ProteinNet.__len__(self)

class DebugLoss(nn.Module):
  def __init__(self):
    super().__init__()
    self.loss = nn.CrossEntropyLoss()

  def forward(self, inputs, targets):
    return self.loss(inputs, targets)

class StructuredTransformerTraining(SupervisedTraining):
  def each_step(self):
    learning_rate = torch.pow(torch.tensor(128.0), -0.5)
    step_num = torch.tensor(float(self.step_id + 1))
    learning_rate *= min(
      torch.pow(step_num, -0.5),
      step_num * torch.pow(torch.tensor(4000.0), -1.5)
    )
    self.optimizer.param_groups[0]["lr"] = learning_rate

if __name__ == "__main__":
  num_neighbours = 30 if len(sys.argv) < 4 else int(sys.argv[3])
  data = TransformerNet(sys.argv[1], num_neighbours=num_neighbours)
  valid_data = TransformerNet(sys.argv[2], num_neighbours=num_neighbours)
  net = FlexibleTransformer(
    6, 20, 128, 10, 64,
    attention_size=128, heads=8,
    mlp_depth=2, depth=3, batch_norm=True
  )
  training = StructuredTransformerTraining(
    net, data, valid_data,
    [DebugLoss()],
    batch_size=8,
    max_epochs=1000,
    optimizer=lambda x: torch.optim.Adam(x, lr=1e-5),
    device="cuda:0",
    network_name="structured-transformer-long",
    valid_callback=valid_callback
  ).load()
  final_net = training.train()
