import sys
import random

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as func
from torch.nn.utils.spectral_norm import spectral_norm
from torchsupport.training.energy import EnergyTraining
from torchsupport.training.samplers import PackedLangevin, Langevin, ResetLangevin, SmartResetLangevin

from torchsupport.modules.basic import MLP
from torchsupport.data.io import to_device
from torchsupport.structured import PackedTensor, ConstantStructure, SubgraphStructure
from torchsupport.structured import DataParallel as SDP
from torchsupport.structured import scatter

from protsupport.data.proteinnet import ProteinNet, ProteinNetKNN
from protsupport.utils.geometry import orientation
from protsupport.modules.structures import RelativeStructure
from protsupport.modules.structured_energy import StructuredEnergy, StupidEnergy, DistanceStupidEnergy, ProperEnergy, GaussianStupidEnergy, DilatedStupidEnergy
from protsupport.modules.anglespace import PositionLookup
from protsupport.modules.backrub import Backrub
from protsupport.modules.transformer import attention_connected, linear_connected, assignment_connected

from torchsupport.optim.diffmod import DiffMod

AA_CODE = "ACDEFGHIKLMNPQRSTVWY"

class EBMNet(ProteinNetKNN):
  def __init__(self, path, num_neighbours=20, n_jobs=1, N=64, cache=True):
    ProteinNetKNN.__init__(
      self, path,
      num_neighbours=num_neighbours,
      n_jobs=n_jobs, cache=cache
    )
    self.backrub = Backrub(n_moves=0)
    self.ors = torch.tensor(
      orientation(self.ters[1].numpy() / 100).transpose(2, 0, 1),
      dtype=torch.float
    )
    self.N = N
    self.valid_indices = [
      index
      for index in range(len(self.index) - 1)
      if self.index[index + 1] - self.index[index] >= N
    ]

  def __getitem__(self, index):
    N = self.N
    index = self.valid_indices[index]
    window = slice(self.index[index], min(self.index[index + 1], self.index[index] + N))
    inds = self.inds[window]
    primary = self.pris[window] - 1

    # add noise:
    n_positions = random.randrange(max(1, primary.size(0) // 5))
    primary[torch.randint(0, primary.size(0), (n_positions,))] = torch.randint(0, 20, (n_positions,))

    tertiary = self.ters[:, :, window]
    distances, angles = self.backrub(tertiary[[0, 1, 3]].permute(2, 0, 1))
    distances = distances[:, 1] / 100

    distances = (distances[None, :] - distances[:, None]).norm(dim=-1).unsqueeze(0) / 100
    distances = 0.99 * distances + 0.01 * torch.rand_like(distances)
    sequence = torch.zeros(42, N, N)
    sequence[-1] = 1

    protein = SubgraphStructure(torch.zeros(distances.size(0), dtype=torch.long))
    neighbours = ConstantStructure(0, 0, (inds - self.index[index]).to(torch.long))

    primary_onehot = torch.zeros(primary.size(0), 20, dtype=torch.float)
    primary_onehot[torch.arange(primary.size(0)), primary] = 1
    primary_onehot = primary_onehot.clamp(0, 1)
#    sequence[:20, :, :] = primary_onehot.permute(1, 0)[:, None, :]
#    sequence[20:40, :, :] = primary_onehot.permute(1, 0)[:, :, None]
#
#    mask = torch.rand(primary.size(0)) < torch.rand(1)
#
#    sequence[:, mask, :] = 0
#    sequence[-1, mask, :] = 1
#    sequence[:, :, mask] = 0
#    sequence[-1, :, mask] = 1

    #assert neighbours.connections.max() < primary_onehot.size(0)
    inputs = (
      distances, sequence
    )

    return inputs

  def __len__(self):
    return len(self.valid_indices)

class EBMTraining(EnergyTraining):
  def prepare(self):
    index = random.randrange(0, len(self.data))
    (distance, ground_truth) = self.data[index]
    distance = torch.rand_like(distance)
    distance = (distance + distance.permute(0, 2, 1)) / 2
    distance[:, torch.arange(distance.size(-1)), torch.arange(distance.size(-1))] = 0
    return (
      distance, ground_truth
    )

  #def decompose_batch(self, data, *args):
  #  count = len(data)
  #  targets = [self.device] * count
  #  gt, protein = args
  #  protein = protein.chunk(targets)
  #  result = [
  #    to_device((
  #      data[idx].detach(),
  #      gt[idx].detach(),
  #      protein[idx]
  #    ), "cpu")
  #    for idx in range(count)
  #  ]
#
#    return result

  def each_generate(self, data, gt):
    with torch.no_grad():
      dist = data.cpu()
      dist = (dist + dist.permute(0, 1, 3, 2)) / 2
      dist[:, :, torch.arange(dist.size(-1)), torch.arange(dist.size(-1))] = 0
      dist = dist.numpy()
      fig, ax = plt.subplots(figsize=(10, 10))
      ax.matshow(dist[0, 0])
      self.writer.add_figure("dist", fig, self.step_id)

class DDP(nn.Module):
  def __init__(self, net):
    super().__init__()
    self.net = net

  def forward(self, *args):
    inputs = []
    for arg in args:
      if isinstance(arg, PackedTensor):
        inputs.append(arg.tensor)
      else:
        inputs.append(arg)
    return self.net(*inputs)

if __name__ == "__main__":
  data = EBMNet(sys.argv[1], num_neighbours=15)
  net = SDP(
    ProperEnergy(depth=16, down=4, shape=64)
  )
  integrator = Langevin(rate=50.0, noise=0.01, steps=20, max_norm=None, clamp=(0, 1))
  training = EBMTraining(
    net, data,
    batch_size=32,
    decay=1.0,
    max_epochs=5000,
    integrator=integrator,
    buffer_probability=0.95,
    buffer_size=10000,
    optimizer_kwargs={"lr": 5e-5, "betas": (0.0, 0.999)},
    device="cuda:0",
    network_name="distance-ebm/stoopid-ebm-distance-zeroed-spatial-very-large-54",
    verbose=True
  ).load()
  final_net = training.train()
