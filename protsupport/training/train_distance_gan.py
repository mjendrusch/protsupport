import sys
import random

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import torch
import torch.nn as nn
import torch.nn.functional as func
from torch.nn.utils.spectral_norm import spectral_norm
from torchsupport.training.gan import RothGANTraining
from torchsupport.optim.diffmod import DiffMod

from torchsupport.modules.basic import MLP
from torchsupport.data.io import to_device, make_differentiable
from torchsupport.structured import PackedTensor, ConstantStructure, SubgraphStructure
from torchsupport.structured import DataParallel as SDP
from torchsupport.structured import scatter

from protsupport.data.proteinnet import ProteinNet, ProteinNetKNN
from protsupport.utils.geometry import orientation
from protsupport.modules.structures import RelativeStructure
from protsupport.modules.distance_graph_gan import DistanceGenerator, DistanceDiscriminator
from protsupport.modules.anglespace import PositionLookup
from protsupport.modules.backrub import Backrub
from protsupport.modules.transformer import attention_connected, linear_connected, assignment_connected

AA_CODE = "ACDEFGHIKLMNPQRSTVWY"

class GANNet(ProteinNetKNN):
  def __init__(self, path, num_neighbours=20, n_jobs=1, cache=True):
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

  def __getitem__(self, index):
    length = torch.randint(32, 65, (1,))[0]
    window = slice(self.index[index], min(self.index[index + 1], self.index[index] + length))
    inds = self.inds[window]
    primary = self.pris[window] - 1

    # add noise:
    n_positions = random.randrange(max(1, primary.size(0) // 100))
    primary[torch.randint(0, primary.size(0), (n_positions,))] = torch.randint(0, 20, (n_positions,))

    tertiary = self.ters[:, :, window]
    tertiary, angles = self.backrub(tertiary[[0, 1, 3]].permute(2, 0, 1))
    tertiary = tertiary[:, 1] / 100

    protein = SubgraphStructure(torch.zeros(tertiary.size(0), dtype=torch.long))
    #neighbours = ConstantStructure(0, 0, (inds - self.index[index]).to(torch.long))
    distances, _ = scatter.pairwise_no_pad(lambda x, y: (x - y).norm(dim=1), tertiary, protein.indices)
    distances = distances[:, None]

    primary_onehot = torch.zeros(primary.size(0), 20, dtype=torch.float)
    primary_onehot[torch.arange(primary.size(0)), primary] = 1
    primary_onehot = primary_onehot.clamp(0, 1)

    #assert neighbours.connections.max() < primary_onehot.size(0)
    inputs = (
      PackedTensor(angles.permute(1, 0).contiguous()),
      PackedTensor(distances),
      protein
    )

    return inputs

  def __len__(self):
    return ProteinNet.__len__(self)

class AngleGANTraining(RothGANTraining):
  def mixing_key(self, data):
    return data[0].tensor

  def each_generate(self, data, generated, sample):
    with torch.no_grad():
      angles, distances, protein = generated
      angs = angles.tensor.cpu()
      dists = distances.tensor.cpu()
      first_length = (protein.indices.cpu() == 0).sum()
      expanded = first_length * (first_length - 1) // 2
      first_dist = dists[:expanded]
      dist = torch.zeros(first_length, first_length)
      count = 0
      for idx in range(first_length):
        for idy in range(idx + 1, first_length):
          dist[idx, idy] = dists[count]
          dist[idy, idx] = dists[count]
          count += 1

      lookup = PositionLookup()
      c_alpha, _ = lookup(angs[protein.indices.cpu() == 0], torch.zeros_like(protein.indices[protein.indices == 0].cpu()))
      c_alpha = c_alpha[:, 1]
      c_alpha = c_alpha.numpy()
      dist = dist.numpy()
      fig = plt.figure()
      ax = fig.add_subplot(111, projection='3d')
      ax.plot(c_alpha[:, 0], c_alpha[:, 1], c_alpha[:, 2])
      self.writer.add_figure("output", fig, self.step_id)
      fig = plt.figure()
      ax = fig.add_subplot(111)
      ax.scatter(angs[:, 1], angs[:, 2])
      self.writer.add_figure("rama", fig, self.step_id)
      fig, ax = plt.subplots()
      ax.matshow(dist)
      self.writer.add_figure("dist", fig, self.step_id)

class DDP(nn.Module):
  def __init__(self, net):
    super().__init__()
    self.net = net

  def sample(self, *args, **kwargs):
    return self.net.sample(*args, **kwargs)

  def forward(self, *args):
    inputs = []
    print(args)
    for arg in args:
      print(type(arg))
      if isinstance(arg, PackedTensor):
        print("yay")
        inputs.append(arg.tensor)
      else:
        inputs.append(arg)
    return self.net(*inputs)

torch.backends.cudnn.enabled = False

if __name__ == "__main__":
  data = GANNet(sys.argv[1], num_neighbours=15)
  gen = SDP(
    DistanceGenerator(128, depth=3)
  )
  disc = SDP(
    DistanceDiscriminator(depth=3)
  )
  training = AngleGANTraining(
    gen, disc, data,
    batch_size=8,
    max_epochs=1000,
    #optimizer=DiffMod,
    device="cuda:0",
    network_name="distance-gan/distribution-fix",
    verbose=True,
    report_interval=10
  )
  final_net = training.train()
