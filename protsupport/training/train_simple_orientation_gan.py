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
from protsupport.utils.geometry import orientation, relative_orientation
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
    self.valid_indices = [
      index
      for index in range(len(self.index) - 1)
      if self.index[index + 1] - self.index[index] >= N
    ]
    self.backrub = Backrub(n_moves=0)
    self.ors = torch.tensor(
      orientation(self.ters[1].numpy() / 100).transpose(2, 0, 1),
      dtype=torch.float
    )

  def orientations(self, tertiary):
    ors = orientation(tertiary.permute(1, 0)).permute(2, 0, 1).contiguous()
    return ors.view(tertiary.size(0), 3, 3)

  def __getitem__(self, index):
    N = self.N
    index = self.valid_indices[index]
    window = slice(self.index[index], min(self.index[index + 1], self.index[index] + N))
    inds = self.inds[window]
    primary = self.pris[window] - 1

    # add noise:
    n_positions = random.randrange(max(1, primary.size(0) // 100))
    primary[torch.randint(0, primary.size(0), (n_positions,))] = torch.randint(0, 20, (n_positions,))

    tertiary = self.ters[:, :, window]
    tertiary, angles = self.backrub(tertiary[[0, 1, 3]].permute(2, 0, 1))
    tertiary = tertiary[:, 1] / 100

    ors = self.orientations(tertiary)
    print(ors.shape)

    count = tertiary.size(0)

    x_range = torch.repeat_interleave(torch.arange(count), count * torch.ones(count, dtype=torch.long))
    y_range = torch.arange(count ** 2) - x_range * count
    t_x = tertiary[x_range]
    t_y = tertiary[y_range]
    o_x = ors[x_range]
    o_y = ors[y_range]
    _, directions, rotations = relative_orientation(t_x, t_y, o_x, o_y)
    directions = directions.reshape(count, count, *directions.shape[1:]).permute(2, 0, 1).contiguous()
    rotations = rotations.reshape(count, count, *rotations.shape[1:]).permute(2, 0, 1).contiguous()

    mask = torch.arange(N)
    mask = (mask[:, None] - mask[None, :]) > 0
    mask = mask.float()

    directions = mask * directions + (1 - mask) * directions.permute(0, 2, 1).contiguous()
    rotations = mask * rotations + (1 - mask) * rotations.permute(0, 2, 1).contiguous()

    protein = SubgraphStructure(torch.zeros(tertiary.size(0), dtype=torch.long))
    #neighbours = ConstantStructure(0, 0, (inds - self.index[index]).to(torch.long))
    #distances, _ = scatter.pairwise_no_pad(lambda x, y: (x - y).norm(dim=1), tertiary, protein.indices)
    distances = (tertiary[None, :, :] - tertiary[:, None, :]).norm(dim=-1)
    distances = distances.unsqueeze(0)

    primary_onehot = torch.zeros(primary.size(0), 20, dtype=torch.float)
    primary_onehot[torch.arange(primary.size(0)), primary] = 1
    primary_onehot = primary_onehot.clamp(0, 1)

    result = distances / 100
    result = torch.cat((result, rotations, directions), dim=0)

    return (result,)

  def __len__(self):
    return len(self.valid_indices)

class AngleGANTraining(RothGANTraining):
  def mixing_key(self, data):
    return data[0]

  def each_generate(self, data, generated, sample):
    with torch.no_grad():
      features, *_ = generated
      distances = features[:, :1, :, :]
      rotations = features[:, 1:5, :, :]
      directions = features[:, 5:, :, :]
      dist = distances.cpu()
      rots = rotations.cpu()
      dirs = directions.cpu()
      
      real, *_ = data
      rdist = real[:, :1, :, :].detach().cpu()
      rrots = real[:, 1:5, :, :].detach().cpu()
      rdirs = real[:, 5:, :, :].detach().cpu()

      dist = dist.numpy()
      fig, ax = plt.subplots()
      ax.matshow(dist[0, 0])
      self.writer.add_figure("dist", fig, self.step_id)

      rots = (rots[0, 1:].permute(1, 2, 0) + 1) / 2
      rots = rots.numpy()

      fig, ax = plt.subplots()
      ax.matshow(rots)
      self.writer.add_figure("rots", fig, self.step_id)

      dirs = (dirs[0].permute(1, 2, 0) + 1) / 2
      dirs = dirs.numpy()

      fig, ax = plt.subplots()
      ax.matshow(dirs)
      self.writer.add_figure("dirs", fig, self.step_id)

      rdist = rdist.numpy()
      fig, ax = plt.subplots()
      ax.matshow(rdist[0, 0])
      self.writer.add_figure("real dist", fig, self.step_id)

      rrots = (rrots[0, 1:].permute(1, 2, 0) + 1) / 2
      rrots = rrots.numpy()

      fig, ax = plt.subplots()
      ax.matshow(rrots)
      self.writer.add_figure("real rots", fig, self.step_id)

      rdirs = (rdirs[0].permute(1, 2, 0) + 1) / 2
      rdirs = rdirs.numpy()

      fig, ax = plt.subplots()
      ax.matshow(rdirs)
      self.writer.add_figure("real dirs", fig, self.step_id)

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

def rotate_quaternion(q, v):
  shape = list(v.shape)
  q = q.view(-1, 4)
  v = v.view(-1, 3)

  qvec = q[:, 1:]
  uv = torch.cross(qvec, v, dim=1)
  uuv = torch.cross(qvec, uv, dim=1)
  return (v + 2 * (q[:, :1] * uv + uuv)).view(shape)

class Predictor2d(nn.Module):
  def __init__(self, in_size, out_size):
    super().__init__()
    self.blocks = nn.ModuleList([
      nn.Sequential(
        nn.Conv2d(in_size, in_size, 3, dilation=2 ** (idx % 5), padding=2 ** (idx % 5)),
        nn.InstanceNorm2d(in_size),
        nn.LeakyReLU(),
      )
      for idx in range(10)
    ])
    self.predict = nn.Conv2d(in_size, out_size, 1)

  def forward(self, inputs):
    out = inputs
    for block in self.blocks:
      out = out + block(out)
    return self.predict(out)

class OrientationGenerator(nn.Module):
  def __init__(self, depth=4):
    super().__init__()
    width = 256
    self.preprocess = nn.Linear(1024, 512 * 4 * 4, bias=True)
    self.blocks = nn.ModuleList([
      nn.Sequential(
        nn.Conv2d(512 + 2, 256 * 4, 3, padding=1),
        nn.InstanceNorm2d(256 * 4),
        nn.LeakyReLU()
      ),
      nn.Sequential(
        nn.Conv2d(256 + 2, width * 4, 3, padding=1),
        nn.InstanceNorm2d(width * 4),
        nn.LeakyReLU()
      )
    ] + [
      nn.Sequential(
        nn.Conv2d(width + 2, width * 4, 3, padding=1),
        nn.InstanceNorm2d(width * 4),
        nn.LeakyReLU()
      )
      for idx in range(depth - 2)
    ])
    self.distances = Predictor2d(width, 1)#nn.Conv2d(128, 1, 3, padding=1)
    self.direction = Predictor2d(width, 3)#nn.Conv2d(128, 3, 3, padding=1)
    self.rotation = Predictor2d(width, 4)#nn.Conv2d(128, 4, 3, padding=1)

  def sample(self, batch_size):
    return torch.randn(batch_size, 1024)

  def predict_rotation(self, out):
    raw_quaternion = self.rotation(out)
    raw_quaternion_sym = (raw_quaternion[:, :1] + raw_quaternion[:, :1].permute(0, 1, 3, 2)) / 2
    raw_quaternion_anti = (raw_quaternion[:, 1:] - raw_quaternion[:, 1:].permute(0, 1, 3, 2)) / 2
    raw_quaternion = torch.cat((raw_quaternion_sym, raw_quaternion_anti), dim=1)
    raw_quaternion = raw_quaternion / raw_quaternion.norm(dim=1, keepdim=True).detach()
    print(raw_quaternion.shape)
    #inverse_quaternion = torch.cat((raw_quaternion[:, :1], -raw_quaternion[:, 1:]), dim=1)
    
    # upper triangular mask
    mask = torch.arange(out.size(-1), device=out.device)
    mask = (mask[:, None] - mask[None, :]) > 0
    mask = mask.float()

    ind = torch.arange(raw_quaternion.size(-1), device=out.device)
    rotation = raw_quaternion# * mask + inverse_quaternion * (1 - mask)
    rotation[:, :, ind, ind] = 0
    rotation[:, 0, ind, ind] = 1

    raw_direction = self.direction(out)
    antisymmetric = raw_direction - raw_direction.permute(0, 1, 3, 2)
    unit = antisymmetric / (antisymmetric.norm(dim=1, keepdim=True).detach() + 1e-12)

    direction = unit.clone()
    vshape = direction.permute(0, 2, 3, 1).shape
    vdirections = direction.permute(0, 2, 3, 1).reshape(-1, 3)
    vdirections = rotate_quaternion(rotation.permute(0, 3, 2, 1).reshape(-1, 4), vdirections)
    vdirections = vdirections.reshape(*vshape).permute(0, 3, 1, 2)
    direction = mask * direction + (1 - mask) * vdirections.reshape(*direction.shape) 
    direction[:, :, ind, ind] = 0

    return rotation, direction

  def forward(self, sample):
    out = self.preprocess(sample).view(-1, 512, 4, 4)
    for idx, block in enumerate(self.blocks):
      pos = torch.arange(out.size(-1), dtype=out.dtype, device=out.device) / 100
      pos = pos[None].expand(out.size(0), out.size(-1))
      sym = (pos[:, None, :] + pos[:, :, None]) / 2
      asym = (pos[:, None, :] - pos[:, :, None]) / 2
      combined = torch.cat((out, sym[:, None], asym[:, None]), dim=1)
      out = block(combined)
      out = func.pixel_shuffle(out, 2)
      #out = func.interpolate(out, scale_factor=2)
    mask = torch.arange(out.size(-1), device=out.device)
    mask = (mask[:, None] - mask[None, :]) > 0
    mask = mask.float()
    distances = func.softplus(self.distances(out))
    distances = (distances + distances.permute(0, 1, 3, 2)) / 2
    rotation = self.rotation(out)
    rotation = mask[None, None] * rotation + (1 - mask[None, None]) * rotation.permute(0, 1, 3, 2)
    #rotation = rotation + rotation.permute(0, 1, 3, 2)
    rotation = rotation.sin() / (rotation.sin().norm(dim=1, keepdim=True).detach() + 1e-6)
    direction = self.direction(out)
    direction = mask[None, None] * direction.permute(0, 1, 3, 2) + (1 - mask[None, None]) * direction
    #direction = direction + direction.permute(0, 1, 3, 2)
    direction = direction.sin() / (direction.sin().norm(dim=1, keepdim=True).detach() + 1e-6)
    #rotation, direction = self.predict_rotation(out)
    size = out.size(-1)
    ind = torch.arange(size, device=out.device)
    distances[:, :, ind, ind] = 0
    out = torch.cat((distances, rotation, direction), dim=1)
    return (out,)

class MaskedGenerator(nn.Module):
  def __init__(self, data, depth=6):
    super().__init__()
    self.data = data
    self.masked_process = nn.Conv2d(4, 64, 3, padding=1)
    self.encoder_blocks = nn.ModuleList([
        nn.Sequential(
          nn.Conv2d(64, 64, 3, padding=1),
          nn.BatchNorm2d(128),
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
    self.postprocess = nn.Conv2d(128, 1, 3, padding=1)

  def sample(self, batch_size):
    assembled, given, g_mask, r_mask, range_mask = random.choice(self.data)
    return torch.randn(batch_size, 512), given, g_mask, r_mask, range_mask

  def forward(self, sample):
    latent, given, g_mask, r_mask, range_mask = sample
    mask = torch.cat((given, g_mask, r_mask, range_mask), dim=1)
    shortcut = []
    out = self.masked_process(mask)
    shortcut.append(out)
    for block in self.blocks:
      out = func.avg_pool2d(out, 2)
      out = block(out)
      shortcut.append(out)

    out = self.preprocess(latent).view(-1, 128, 4, 4)
    for idx, block in enumerate(self.blocks):
      combined = torch.cat((out, shortcut[-idx]), dim=1)
      out = block(combined)
      out = func.interpolate(out, scale_factor=2)
    combined = torch.cat((out, shortcut[0]), dim=1)
    out = func.softplus(self.postprocess(out))
    out = (out + out.permute(0, 1, 3, 2)) / 2
    size = out.size(-1)
    out[:, :, torch.arange(size), torch.arange(size)] = 0
    out = r_mask * out + g_mask * given
    return (out, given, g_mask, r_mask, range_mask)

class Discriminator(nn.Module):
  def __init__(self, depth=4):
    super().__init__()
    self.preprocess = nn.Conv2d(8, 128, 5, padding=2)
    self.blocks = nn.ModuleList([
      nn.Sequential(
        nn.Conv2d(128, 128, 3, padding=1),
        nn.InstanceNorm2d(128),
        nn.LeakyReLU(),
        nn.Conv2d(128, 128, 3, dilation=2, padding=2),
        nn.InstanceNorm2d(128),
        nn.LeakyReLU()
      )
      for idx in range(depth)
    ])
    self.predict = nn.Linear(128, 1)

  def forward(self, inputs):
    inputs, *_ = inputs
    out = self.preprocess(inputs)
    for block in self.blocks:
      out = block(out)
      out = func.max_pool2d(out, 2)
    out = func.adaptive_avg_pool2d(out, 1).view(-1, 128)
    out = self.predict(func.dropout(out, 0.5))
    return out

class MaskedDiscriminator(nn.Module):
  def __init__(self):
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
      for idx in range(6)
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
  data = GANNet(sys.argv[1], num_neighbours=15, N=64)
  gen = SDP(
    OrientationGenerator()
  )
  disc = SDP(
    Discriminator()
  )
  training = AngleGANTraining(
    gen, disc, data,
    batch_size=16,
    max_epochs=2000,
    #optimizer=DiffMod,
    device="cuda:0",
    network_name="distance-gan/experiment-orientation-toss-sin-fix",
    verbose=True,
    report_interval=10,
  )
  final_net = training.train()
