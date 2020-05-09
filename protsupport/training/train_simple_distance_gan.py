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
from torchsupport.modules.generative import StyleGANBlock
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
    N = 256
    while self.index[index + 1] - self.index[index] < N:
      index += 1
      index = index % (len(self.index) - 1)
    window = slice(self.index[index], min(self.index[index + 1], self.index[index] + N))
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
    #distances, _ = scatter.pairwise_no_pad(lambda x, y: (x - y).norm(dim=1), tertiary, protein.indices)
    distances = (tertiary[None, :, :] - tertiary[:, None, :]).norm(dim=-1)
    distances = distances.unsqueeze(0)

    primary_onehot = torch.zeros(primary.size(0), 20, dtype=torch.float)
    primary_onehot[torch.arange(primary.size(0)), primary] = 1
    primary_onehot = primary_onehot.clamp(0, 1)

    return (distances,)

  def __len__(self):
    return ProteinNet.__len__(self)

class AngleGANTraining(RothGANTraining):
  def mixing_key(self, data):
    return data[0]

  def each_generate(self, data, generated, sample):
    with torch.no_grad():
      distances, *_ = generated
      dist = distances.cpu()

      dist = dist.numpy()
      fig, ax = plt.subplots()
      ax.matshow(dist[0, 0])
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

class Generator(nn.Module):
  def __init__(self):
    super().__init__()
    self.preprocess = nn.Linear(512, 128 * 4 * 4)
    self.blocks = nn.ModuleList([
      nn.Sequential(
        nn.Conv2d(128, 64, 3, padding=1),
        nn.InstanceNorm2d(64),
        nn.Conv2d(64, 128, 3, padding=1),
        nn.InstanceNorm2d(128),
        nn.ReLU()
      )
      for idx in range(6)
    ])
    self.postprocess = nn.Sequential(
      nn.Conv2d(128, 64, 3, padding=1),
      nn.InstanceNorm2d(64),
      nn.Conv2d(64, 1, 3, padding=1)
    )

  def sample(self, batch_size):
    return torch.randn(batch_size, 512)

  def forward(self, sample):
    out = self.preprocess(sample).view(-1, 128, 4, 4)
    for block in self.blocks:
      out = out + block(out)
      out = func.interpolate(out, scale_factor=2, mode="bilinear")
      #out = out + 0.1 * torch.randn_like(out)
    out = func.softplus(self.postprocess(out))
    out = (out + out.permute(0, 1, 3, 2)) / 2
    out[:, :, torch.arange(256), torch.arange(256)] = 0
    return (out,)

class StyleGenerator(nn.Module):
  def __init__(self):
    super().__init__()
    self.mapping = MLP(512, 512, hidden_size=128, depth=8, batch_norm=False)
    start_features = 512
    self.blocks = nn.ModuleList([
      StyleGANBlock(
        start_features,
        128,
        start_features,
        size=(4, 4),
        activation=func.relu_
      )
    ] + [
      StyleGANBlock(
        128,
        128,
        start_features,
        activation=func.relu_
      )
      for idx in range(6)
    ])
    self.postprocess = nn.Conv2d(128, 1, 3, padding=1)

  def sample(self, batch_size):
    return torch.randn(batch_size, 512)

  def forward(self, sample):
    cond = self.mapping(sample)
    out = cond
    for block in self.blocks:
      out = block(out, cond)
    out = func.softplus(self.postprocess(out))
    out = (out + out.permute(0, 1, 3, 2)) / 2
    out[:, :, torch.arange(256), torch.arange(256)] = 0
    return (out,)

class StupidGenerator(nn.Module):
  def __init__(self):
    super().__init__()
    self.preprocess = nn.Conv2d(512, 128, 3, padding=1)
    self.blocks = nn.ModuleList([
      nn.Sequential(
        nn.Conv2d(128, 128, 3, padding=1),
        nn.InstanceNorm2d(128),
        nn.ReLU()
      )
      for idx in range(4)
    ])
    self.postprocess = nn.Conv2d(128, 1, 3, padding=1)

  def sample(self, batch_size):
    return torch.randn(batch_size, 512, 64, 64)

  def forward(self, sample):
    out = self.preprocess(sample)
    for idx, block in enumerate(self.blocks):
      out = out + func.interpolate(block(func.avg_pool2d(out, 2 ** (4 - idx))), scale_factor=2 ** (4 - idx), mode="bilinear")
    out = func.softplus(self.postprocess(out))
    out = (out + out.permute(0, 1, 3, 2)) / 2
    out[:, :, torch.arange(64), torch.arange(64)] = 0
    return (out,)

class Discriminator(nn.Module):
  def __init__(self):
    super().__init__()
    self.preprocess = nn.Conv2d(1, 128, 5, padding=2)
    self.blocks = nn.ModuleList([
      nn.Sequential(
        nn.Conv2d(128, 128, 3, padding=1),
        nn.InstanceNorm2d(128),
        nn.ReLU(),
        nn.Conv2d(128, 128, 3, dilation=2, padding=2),
        nn.InstanceNorm2d(128),
        nn.ReLU()
      )
      for idx in range(6)
    ])
    self.predict = nn.Linear(128, 1)

  def forward(self, inputs):
    inputs, *_ = inputs
    out = self.preprocess(inputs)
    for block in self.blocks:
      out = out + block(out)
      out = func.max_pool2d(out, 2)
    out = func.adaptive_avg_pool2d(out, 1).view(-1, 128)
    out = self.predict(out)
    return out

if __name__ == "__main__":
  data = GANNet(sys.argv[1], num_neighbours=15)
  gen = SDP(
    StyleGenerator()
  )
  disc = SDP(
    Discriminator()
  )
  training = AngleGANTraining(
    gen, disc, data,
    batch_size=16,
    max_epochs=1000,
    #optimizer=DiffMod,
    device="cuda:0",
    network_name="distance-gan/simple-as-fuck-style-1",
    verbose=True,
    report_interval=10
  )
  final_net = training.train()
