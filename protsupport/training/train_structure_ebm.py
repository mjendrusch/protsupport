import sys
import random

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import torch
import torch.nn as nn
import torch.nn.functional as func
from torch.nn.utils.spectral_norm import spectral_norm
from torchsupport.training.energy import EnergyTraining
from torchsupport.training.samplers import PackedLangevin

from torchsupport.modules.basic import MLP
from torchsupport.data.io import to_device
from torchsupport.structured import PackedTensor, ConstantStructure, SubgraphStructure
from torchsupport.structured import DataParallel as SDP
from torchsupport.structured import scatter

from protsupport.data.proteinnet import ProteinNet, ProteinNetKNN
from protsupport.utils.geometry import orientation
from protsupport.modules.structures import RelativeStructure
from protsupport.modules.structured_energy import StructuredEnergy
from protsupport.modules.anglespace import PositionLookup

AA_CODE = "ACDEFGHIKLMNPQRSTVWY"

class EBMNet(ProteinNetKNN):
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

    # add noise:
    n_positions = random.randrange(max(1, primary.size(0) // 100))
    primary[torch.randint(0, primary.size(0), (n_positions,))] = torch.randint(0, 20, (n_positions,))

    tertiary = self.ters[:, :, window]
    distances = tertiary.permute(2, 0, 1) / 100

    protein = SubgraphStructure(torch.zeros(distances.size(0), dtype=torch.long))
    neighbours = ConstantStructure(0, 0, (inds - self.index[index]).to(torch.long))

    primary_onehot = torch.zeros(primary.size(0), 20, dtype=torch.float)
    primary_onehot[torch.arange(primary.size(0)), primary] = 1
    primary_onehot = primary_onehot.clamp(0, 1)


    assert neighbours.connections.max() < primary_onehot.size(0)
    inputs = (
      PackedTensor(distances),
      PackedTensor(primary_onehot),
      protein
    )

    return inputs

  def __len__(self):
    return ProteinNet.__len__(self)

class EBMTraining(EnergyTraining):
  def prepare(self):
    position_lookup = PositionLookup()
    index = random.randrange(0, len(self.data))
    (positions, ground_truth, protein) = self.data[index]
    angles = torch.randn(positions.tensor.size(0), 3)
    positions, _ = position_lookup(angles, protein.indices)
    return (
      PackedTensor(positions), ground_truth, protein
    )

  def decompose_batch(self, data, *args):
    count = len(data)
    targets = [self.device] * count
    gt, protein = args
    #gt = gt.chunk(targets)
    protein = protein.chunk(targets)
    result = [
      to_device((
        data[idx].detach(),
        gt[idx].detach(),
        protein[idx]
      ), "cpu")
      for idx in range(count)
    ]

    return result

  def each_generate(self, data, gt, protein):
    with torch.no_grad():
      c_alpha = data.tensor[protein.indices == 0, 1].numpy()
      fig = plt.figure()
      ax = fig.add_subplot(111, projection='3d')
      ax.plot(c_alpha[:, 0], c_alpha[:, 1], c_alpha[:, 2])
      self.writer.add_figure("output", fig, self.step_id)

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
    StructuredEnergy(
      6, 128, 10, 
      attention_size=128, heads=8,
      mlp_depth=2, depth=3, batch_norm=True,
      neighbours=15
    )
  )
  integrator = PackedLangevin(rate=50, noise=0.01, steps=20, max_norm=None, clamp=None)
  training = EBMTraining(
    net, data,
    batch_size=4,
    decay=1.0,
    max_epochs=1000,
    integrator=integrator,
    buffer_probability=0.95,
    buffer_size=10000,
    optimizer_kwargs={"lr": 1e-4},
    device="cuda:0",
    network_name="distance-gan/pooled-ebm-1",
    verbose=True
  )
  final_net = training.train()
