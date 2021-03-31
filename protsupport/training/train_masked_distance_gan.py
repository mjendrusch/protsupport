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
  def __init__(self, path, num_neighbours=20, N=256, n_jobs=1, cache=True):
    ProteinNetKNN.__init__(
      self, path,
      num_neighbours=num_neighbours,
      n_jobs=n_jobs, cache=cache
    )
    self.N = N
    self.backrub = Backrub(n_moves=0)
    self.ors = torch.tensor(
      orientation(self.ters[1].numpy() / 100).transpose(2, 0, 1),
      dtype=torch.float
    )

  def __getitem__(self, index):
    N = self.N
    window = slice(self.index[index], min(self.index[index + 1], self.index[index] + N))

    item_length = min(self.index[index + 1] - self.index[index], N)

    # construct placement mask
    placement = 0 if item_length == N else random.randrange(0, N - item_length)
    range_mask = torch.zeros(N, dtype=torch.uint8)
    range_mask[placement:placement + item_length] = 1
    range_mask = range_mask[None, :] * range_mask[:, None]
    range_mask = range_mask.unsqueeze(0)

    # construct available / requested mask
    requested_length = random.randrange(1, item_length)
    requested_placement = placement + random.randrange(0, item_length - requested_length)
    available_mask = range_mask.clone()
    available_mask[:, requested_placement:requested_placement + requested_length, :] = 0
    available_mask[:, :, requested_placement:requested_placement + requested_length] = 0
    requested_mask = (1 - available_mask) * range_mask

    inds = self.inds[window]
    primary = self.pris[window] - 1

    # add noise:
    n_positions = random.randrange(max(1, primary.size(0) // 100))
    primary[torch.randint(0, primary.size(0), (n_positions,))] = torch.randint(0, 20, (n_positions,))

    tertiary = self.ters[:, :, window]
    tertiary, angles = self.backrub(tertiary[[0, 1, 3]].permute(2, 0, 1))
    tertiary = tertiary[:, 1] / 100

    raw_distances = (tertiary[None, :, :] - tertiary[:, None, :]).norm(dim=-1)
    raw_distances = raw_distances.unsqueeze(0)

    distances = torch.zeros(1, N, N)
    distances[:, placement:placement + item_length, placement:placement + item_length] = raw_distances
    distances = distances / 100

    available = distances * available_mask.float()

    return (distances, available, available_mask.float(), requested_mask.float(), range_mask.float())

  def __len__(self):
    return len(self.index) - 1

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
    self.preprocess = nn.Linear(512, 128 * 4 * 4, bias=False)
    self.blocks = nn.ModuleList([
      nn.Sequential(
        nn.Conv2d(128, 128, 3, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU()
      )
      for idx in range(6)
    ])
    self.postprocess = nn.Conv2d(128, 1, 3, padding=1)

  def sample(self, batch_size):
    return torch.randn(batch_size, 512)

  def forward(self, sample):
    out = self.preprocess(sample).view(-1, 128, 4, 4)
    for idx, block in enumerate(self.blocks):
      out = block(out)
      out = func.interpolate(out, scale_factor=2)
    out = func.softplus(self.postprocess(out))
    out = (out + out.permute(0, 1, 3, 2)) / 2
    size = out.size(-1)
    out[:, :, torch.arange(size), torch.arange(size)] = 0
    return (out,)

class MaskedGenerator(nn.Module):
  def __init__(self, data, depth=6):
    super().__init__()
    self.data = data
    self.masked_process = nn.Conv2d(4, 64, 3, padding=1)
    self.encoder_blocks = nn.ModuleList([
        nn.Sequential(
          nn.Conv2d(64, 64, 3, padding=1),
          nn.BatchNorm2d(64),
          nn.ReLU()
        )
        for idx in range(depth)
    ])

    self.preprocess = nn.Linear(512, 128 * 4 * 4, bias=False)
    self.blocks = nn.ModuleList([
      nn.Sequential(
        nn.Conv2d(128 + 64, 128, 3, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU()
      )
      for idx in range(depth)
    ])
    self.scales = nn.ModuleList([
      nn.Linear(512, 128)
      for idx in range(depth)
    ])
    self.biases = nn.ModuleList([
      nn.Linear(512, 128)
      for idx in range(depth)
    ])
    self.postprocess = nn.Conv2d(128 + 64, 1, 3, padding=1)

  def sample(self, batch_size):
    g = []
    gm = []
    rm = []
    rngm = []
    for idx in range(batch_size):
      assembled, given, g_mask, r_mask, range_mask = random.choice(self.data)
      g.append(given.unsqueeze(0))
      gm.append(g_mask.unsqueeze(0))
      rm.append(r_mask.unsqueeze(0))
      rngm.append(range_mask.unsqueeze(0))
    given = torch.cat(g, dim=0)
    g_mask = torch.cat(gm, dim=0)
    r_mask = torch.cat(rm, dim=0)
    range_mask = torch.cat(rngm, dim=0)
    return torch.randn(batch_size, 512), given, g_mask, r_mask, range_mask

  def forward(self, sample):
    latent, given, g_mask, r_mask, range_mask = sample
    mask = torch.cat((given, g_mask, r_mask, range_mask), dim=1)
    shortcut = []
    out = self.masked_process(mask)
    shortcut.append(out)
    for block in self.encoder_blocks:
      out = func.avg_pool2d(out, 2)
      out = block(out)
      shortcut.append(out)

    out = self.preprocess(latent).view(-1, 128, 4, 4)
    for idx, (block, scale, bias) in enumerate(zip(self.blocks, self.scales, self.biases)):
      combined = torch.cat((out, shortcut[-(idx + 1)]), dim=1)
      #s = scale(latent).view(-1, 128, 1, 1)
      #b = bias(latent).view(-1, 128, 1, 1)
      #out = s *  block(combined) + b
      out = block(combined)
      out = func.interpolate(out, scale_factor=2)
    combined = torch.cat((out, shortcut[0]), dim=1)
    out = func.softplus(self.postprocess(combined))
    out = (out + out.permute(0, 1, 3, 2)) / 2
    size = out.size(-1)
    out[:, :, torch.arange(size), torch.arange(size)] = 0
    out = r_mask * out + g_mask * given
    return (out, given, g_mask, r_mask, range_mask)

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
      out = func.avg_pool2d(out, 2)
    out = func.adaptive_avg_pool2d(out, 1).view(-1, 128)
    out = self.predict(func.dropout(out, 0.5))
    return out

class MaskedDiscriminator(nn.Module):
  def __init__(self, depth=6):
    super().__init__()
    self.preprocess = nn.Conv2d(5, 128, 5, padding=2)
    self.blocks = nn.ModuleList([
      nn.Sequential(
        nn.Conv2d(128, 128, 3, padding=1),
        nn.InstanceNorm2d(128),
        nn.ReLU(),
        nn.Conv2d(128, 128, 3, dilation=2, padding=2),
        nn.InstanceNorm2d(128),
        nn.ReLU()
      )
      for idx in range(depth)
    ])
    self.predict = nn.Linear(128, 1)

  def forward(self, inputs):
    inputs = torch.cat(inputs, dim=1)
    out = self.preprocess(inputs)
    for block in self.blocks:
      out = out + block(out)
      out = func.avg_pool2d(out, 2)
    out = func.adaptive_avg_pool2d(out, 1).view(-1, 128)
    out = self.predict(func.dropout(out, 0.5))
    return out

if __name__ == "__main__":
  data = GANNet(sys.argv[1], num_neighbours=15, N=128)
  gen = SDP(
    MaskedGenerator(data, depth=5)
  )
  disc = SDP(
    MaskedDiscriminator(depth=5)
  )
  training = AngleGANTraining(
    gen, disc, data,
    batch_size=16,
    max_epochs=1000,
    #optimizer=DiffMod,
    device="cuda:0",
    network_name="distance-gan/maskenschlacht-128-i-am-fucking-stupid-no-modulation-1",
    verbose=True,
    report_interval=10
  )
  final_net = training.train()
