import sys
import random

from matplotlib import pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as func
from torchsupport.training.training import SupervisedTraining
from torchsupport.training.gan import NormalizedDiversityGANTraining
from torchsupport.training.translation import PairedGANTraining

from torchsupport.structured import PackedTensor, ConstantStructure
from torchsupport.structured import DataParallel as SDP

from protsupport.data.proteinnet import ProteinNet, ProteinNetKNN
from protsupport.utils.geometry import orientation
from protsupport.modules.transformer_gan import TransformerGenerator, TransformerDiscriminator

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
      orientation(self.ters[1].numpy() / 100).transpose(2, 0, 1),
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
    distances = self.ters[1, :, window].transpose(0, 1) / 100
    indices = torch.tensor(
      range(window.start, window.stop),
      dtype=torch.float
    )
    indices = indices.view(-1, 1)

    orientation = torch.cat((distances, orientation, indices), dim=1)
    angles = self.angs[:, window].transpose(0, 1)

    neighbours = ConstantStructure(0, 0, (inds - self.index[index]).to(torch.long))

    mask = torch.zeros(primary.size(0))
    number_of_masked = random.randrange(0, primary.size(0))
    mask_positions = torch.randint(0, primary.size(0), (number_of_masked,))
    primary_onehot = torch.zeros(primary.size(0), 20)
    primary_onehot[torch.arange(0, primary.size(0)), primary.view(-1)] = 1
    primary_onehot[mask.nonzero()[0].view(-1)] = 0
    mask[mask_positions] = 1.0

    sin = torch.sin(angles)
    cos = torch.cos(angles)
    angle_features = torch.cat((sin, cos), dim=1)

    features = torch.cat((angle_features, primary_onehot, mask), dim=1)

    inputs = (
      PackedTensor(angle_features),
      PackedTensor(primary_onehot),
      PackedTensor(mask),
      PackedTensor(orientation),
      neighbours
    )

    return inputs, PackedTensor(primary, split=False)

  def __len__(self):
    return ProteinNet.__len__(self)

class DebugLoss(nn.Module):
  def __init__(self):
    super().__init__()
    self.loss = nn.CrossEntropyLoss()

  def forward(self, inputs, targets):
    return self.loss(inputs, targets)

class StructuredTransformerTraining(PairedGANTraining):
  def reconstruction_loss(self, data, generated, sample):
    _, sequence = data
    _, fake_sequence = generated
    loss = func.cross_entropy(fake_sequence, sequence)
    return loss

  # def each_step(self):
  #   learning_rate = torch.pow(torch.tensor(128.0), -0.5)
  #   step_num = torch.tensor(float(self.step_id + 1))
  #   learning_rate *= min(
  #     torch.pow(step_num, -0.5),
  #     step_num * torch.pow(torch.tensor(4000.0), -1.5)
  #   )
  #   self.optimizer.param_groups[0]["lr"] = learning_rate

if __name__ == "__main__":
  data = TransformerNet(sys.argv[1], num_neighbours=15)
  valid_data = TransformerNet(sys.argv[2], num_neighbours=15)
  generator = SDP(TransformerGenerator(
    27, 20, 128, 10, 64,
    attention_size=128, heads=8,
    mlp_depth=2, depth=6, batch_norm=True
  ))
  discriminator = SDP(TransformerDiscriminator(
    27 + 20, 20, 128, 10,
    attention_size=128, heads=8,
    mlp_depth=2, depth=6, batch_norm=True
  ))
  training = StructuredTransformerTraining(
    generator, discriminator, data,
    batch_size=8,
    max_epochs=1000,
    device="cuda:0",
    network_name="structured-transformer-gan"
  ).load()
  final_net = training.train()
