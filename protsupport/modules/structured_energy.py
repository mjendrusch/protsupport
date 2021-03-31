import torch
import torch.nn as nn
import torch.nn.functional as func
from torch.nn.utils.spectral_norm import spectral_norm

from torchsupport.modules.basic import MLP, one_hot_encode
import torchsupport.structured as ts

from protsupport.utils.geometry import relative_orientation
from protsupport.modules.rbf import gaussian_rbf
from protsupport.modules.structures import (
  OrientationStructure, MaskedStructure, RelativeStructure
)
from protsupport.modules.transformer import StructuredTransformerEncoder, StructuredTransformerDecoder
from protsupport.modules.transformer import linear_connected, attention_connected
from protsupport.utils.geometry import orientation
from protsupport.modules.anglespace import PositionLookup

from torchsupport.utils.memory import memory_used

class LocalFeatures(nn.Module):
  def __init__(self, in_size, size):
    super().__init__()
    self.preprocess = spectral_norm(nn.Conv1d(in_size, size, 3, padding=1))
    self.blocks = nn.ModuleList([
      spectral_norm(nn.Conv1d(size, size, 3, dilation=2 ** idx, padding=2 ** idx))
      for idx in range(4)
    ])

  def forward(self, inputs):
    out = func.elu(self.preprocess(inputs))
    for block in self.blocks:
      out = func.elu(block(out))
    return out

class Pool(nn.Module):
  def __init__(self, size, depth):
    super().__init__()
    self.blocks = nn.ModuleList([
      nn.Sequential(
        spectral_norm(nn.Conv1d(size, size, 3, padding=1)),
        nn.LeakyReLU()
      )
      for idx in range(depth)
    ])

  def forward(self, data, indices):
    out, _, _, count = ts.scatter.pad(data, indices)
    out = out.permute(0, 2, 1).contiguous()
    for block in self.blocks:
      out = out + block(out)
      out = func.max_pool1d(out, 2)
    return out.sum(dim=-1)

class StructuredEnergy(nn.Module):
  def __init__(self, in_size, size, distance_size, sequence_size=20,
               attention_size=128, heads=128, hidden_size=128, mlp_depth=3,
               depth=3, max_distance=20, distance_kernels=16, neighbours=15,
               activation=func.relu_, batch_norm=False, conditional=False,
               angles=False, dropout=0.1, connected=linear_connected,
               normalization=lambda x: x):
    super().__init__()
    distance_size = distance_size + distance_kernels - 1
    self.encoder = StructuredTransformerEncoder(
      size, size, distance_size,
      attention_size=attention_size, heads=heads, hidden_size=hidden_size,
      depth=depth, mlp_depth=mlp_depth, activation=activation,
      batch_norm=batch_norm, normalization=spectral_norm, dropout=0.1,
      connected=connected
    )
    self.angles = angles
    self.lookup = PositionLookup()
    self.conditional = conditional
    self.neighbours = neighbours
    self.activation = activation
    self.rbf = (0, max_distance, distance_kernels)
    self.local_features = LocalFeatures(6, size)
    self.reweighting = spectral_norm(nn.Linear(size, 1))
    self.energy = spectral_norm(nn.Linear(size, 1))
    self.pool = Pool(size, 4)

  def orientations(self, tertiary):
    ors = orientation(tertiary[:, 1].permute(1, 0)).permute(2, 0, 1)
    return ors.view(tertiary.size(0), -1)

  def knn_structure(self, tertiary, structure):
    indices = structure.indices
    unique, count = indices.unique(return_counts=True)
    pos = tertiary[:, 1]
    all_neighbours = []
    for index in unique:
      current = pos[structure.indices == index]
      closeness = -(current[:, None] - current[None, :]).norm(dim=-1)
      closeness = closeness + 3 * torch.randn_like(closeness)
      neighbours = closeness.topk(k=self.neighbours, dim=1).indices
      all_neighbours.append(neighbours)
    all_neighbours = torch.cat(all_neighbours, dim=0).to(tertiary.device)
    return ts.ConstantStructure(0, 0, all_neighbours)

  def forward(self, tertiary, sequence, subgraph):
    features = torch.ones(tertiary.size(0), 27, dtype=tertiary.dtype, device=tertiary.device)
    if self.angles:
      angles = tertiary
      asin = angles.sin()
      acos = angles.cos()
      afeat = torch.cat((asin, acos), dim=1)
      features = ts.scatter.batched(self.local_features, afeat, subgraph.indices)
      tertiary, _ = self.lookup(tertiary, torch.zeros_like(subgraph.indices))
    if self.conditional:
      features = sequence
    ors = self.orientations(tertiary)
    pos = tertiary[:, 1]
    inds = torch.arange(0, pos.size(0), dtype=torch.float, device=pos.device).view(-1, 1)
    distances = torch.cat((pos, ors, inds), dim=1)

    structure = self.knn_structure(tertiary, subgraph)

    distance_data = RelativeStructure(structure, self.rbf)
    relative_data = distance_data.message(
      distances, distances
    )
    relative_structure = OrientationStructure(structure, relative_data)

    encoding = self.encoder(features, relative_structure)
    #weight = self.reweighting(encoding)
    #weight = ts.scatter.softmax(weight, subgraph.indices)
    #encoding = ts.scatter.add(weight * encoding, subgraph.indices)
    encoding = self.pool(encoding, subgraph.indices)
    result = self.energy(encoding)

    return result

class StupidEnergy(nn.Module):
  def __init__(self, depth=4, shape=64):
    super().__init__()
    self.preprocess = nn.Conv2d(1, 128, 5, padding=2)
    self.blocks = nn.ModuleList([
      nn.Sequential(
        nn.Conv2d(128, 128, 3, padding=1),
        nn.LeakyReLU(),
        nn.Conv2d(128, 128, 3, dilation=1, padding=1),
        nn.LeakyReLU()
      )
      for idx in range(depth)
    ])
    self.predict = nn.Linear(128, 1)
    self.shape = shape
    self.lookup = PositionLookup()

  def forward(self, angles, sequence, subgraph):
    tertiary, _ = self.lookup(angles, torch.zeros_like(subgraph.indices))
    pos = tertiary[:, 1]
    pos = pos.reshape(-1, self.shape, 3).permute(0, 2, 1)
    angles = angles.reshape(-1, self.shape, 3).permute(0, 2, 1)
    angle_sandwich = torch.cat((
      angles[:, :, None, :].expand(*angles.shape[:2], self.shape, self.shape),
      angles[:, :, :, None].expand(*angles.shape[:2], self.shape, self.shape)
    ), dim=1)
    distances = (pos[:, :, None, :] - pos[:, :, :, None]).norm(dim=1).unsqueeze(1) / 100
    #inputs = torch.cat((
    #  distances, angle_sandwich.sin(), angle_sandwich.cos()
    #), dim=1)
    inputs = distances
    out = self.preprocess(inputs)
    count = 0
    for block in self.blocks:
      out = out + block(out)
      if count % 2 == 0:
        out = func.avg_pool2d(out, 2)
      count += 1
    out = func.adaptive_avg_pool2d(out, 1).view(-1, 128)
    out = self.predict(out)
    return out

class PowerNorm(nn.Module): #FIXME: desperately looking for improvements
  def forward(self, inputs):
    out = inputs.view(inputs.size(0), inputs.size(1), -1)
    out = out / (out ** 2).mean(dim=-1, keepdim=True)
    out = out.reshape(*inputs.shape)
    return out

class DistanceStupidEnergy(nn.Module):
  def __init__(self, depth=4, down=2, shape=64):
    super().__init__()
    self.preprocess = spectral_norm(nn.Conv2d(1 + 1 + 42, 128, 3, padding=1))
    self.blocks = nn.ModuleList([
      nn.Sequential(
        spectral_norm(nn.Conv2d(128, 128, 3, padding=1)),
        nn.LeakyReLU(),
        spectral_norm(nn.Conv2d(128, 128, 3, padding=1)),
        nn.LeakyReLU(),
      )
      for idx in range(depth)
    ])
    self.predict = spectral_norm(nn.Linear(128, 1))
    self.shape = shape
    self.down = down

  def forward(self, distances, sequence):
    ind = torch.arange(distances.size(-1), device=distances.device)
    spatial = abs(ind[None, :].float() - ind[:, None].float())[None, None] / self.shape
    spatial = spatial.expand_as(distances)

    inputs = torch.cat((
      distances, spatial, sequence
    ), dim=1)
    inputs = (inputs + inputs.permute(0, 1, 3, 2)) / 2

    inputs[:, 0, ind, ind] = 0
    out = self.preprocess(inputs)
    count = 0
    for block in self.blocks:
      out = block(out) + out
      if count % self.down == 0:
        out = func.avg_pool2d(out, 2)
      count += 1
    out = func.adaptive_avg_pool2d(out, 1).view(-1, 128)
    out = self.predict(out)
    return out

class ProperEnergy(nn.Module):
  def __init__(self, depth=4, down=2, shape=64):
    super().__init__()
    self.preprocess = spectral_norm(nn.Conv2d(1 + 1 + 42, 128, 3, padding=1))
    self.blocks = nn.ModuleList([
      nn.Sequential(
        spectral_norm(nn.Conv2d(128 * 2 ** (idx // down), 128 * 2 ** (idx // down), 3, padding=1)),
        nn.LeakyReLU(),
        spectral_norm(nn.Conv2d(128 * 2 ** (idx // down), 128 * 2 ** (idx // down), 3, padding=1)),
        nn.LeakyReLU(),
        spectral_norm(nn.Conv2d(128 * 2 ** (idx // down), 128 * 2 ** ((idx + 1) // down), 3, padding=1)),
        nn.LeakyReLU()
      )
      for idx in range(depth)
    ])
    self.predict = spectral_norm(nn.Linear(128 * 2 ** (depth // down), 1))
    self.shape = shape
    self.down = down

  def up(self, tensor, count):
    result = tensor
    if (count + 1) % self.down == 0:
      clone = torch.zeros_like(tensor)
      result = torch.cat((tensor, clone), dim=1)
    return result

  def forward(self, distances, sequence):
    ind = torch.arange(distances.size(-1), device=distances.device)
    spatial = abs(ind[None, :].float() - ind[:, None].float())[None, None] / self.shape
    spatial = spatial.expand_as(distances)

    inputs = torch.cat((
      distances, spatial, sequence
    ), dim=1)
    inputs = (inputs + inputs.permute(0, 1, 3, 2)) / 2

    inputs[:, 0, ind, ind] = 0
    out = self.preprocess(inputs)
    count = 0
    for block in self.blocks:
      out = block(out) + self.up(out, count)
      if count % self.down == 0:
        out = func.avg_pool2d(out, 2)
      count += 1
    out = func.adaptive_avg_pool2d(out, 1).view(out.size(0), -1)
    out = self.predict(out)
    return out

class TertiaryStupidEnergy(nn.Module):
  def __init__(self, depth=4, down=2, shape=64):
    super().__init__()
    self.preprocess = spectral_norm(nn.Conv2d(1 + 1 + 42, 128, 3, padding=1))
    self.blocks = nn.ModuleList([
      nn.Sequential(
        spectral_norm(nn.Conv2d(128, 128, 3, padding=1)),
        nn.LeakyReLU(),
        spectral_norm(nn.Conv2d(128, 128, 3, padding=1)),
        nn.LeakyReLU(),
      )
      for idx in range(depth)
    ])
    self.predict = spectral_norm(nn.Linear(128, 1))
    self.shape = shape
    self.down = down

  def forward(self, tertiary, sequence):
    print(tertiary.shape)
    distances = (tertiary[:, :, None, :] - tertiary[:, :, :, None]).norm(dim=1, keepdim=True) / 100
    ind = torch.arange(distances.size(-1), device=distances.device)
    spatial = abs(ind[None, :].float() - ind[:, None].float())[None, None] / self.shape
    spatial = spatial.expand_as(distances)

    inputs = torch.cat((
      distances, spatial, sequence
    ), dim=1)
    inputs = (inputs + inputs.permute(0, 1, 3, 2)) / 2

    inputs[:, 0, ind, ind] = 0
    out = self.preprocess(inputs)
    count = 0
    for block in self.blocks:
      out = block(out) + out
      if count % self.down == 0:
        out = func.avg_pool2d(out, 2)
      count += 1
    out = func.adaptive_avg_pool2d(out, 1).view(-1, 128)
    out = self.predict(out)
    return out

class DilatedStupidEnergy(nn.Module):
  def __init__(self, depth=4, down=2, shape=64):
    super().__init__()
    self.preprocess = nn.Conv2d(1 + 1 + 42, 128, 5, padding=2)
    self.blocks = nn.ModuleList([
      nn.Sequential(
        spectral_norm(nn.Conv2d(128, 128, 3, padding=1)),
        nn.LeakyReLU(),
        spectral_norm(nn.Conv2d(128, 128, 3, dilation=2 ** (idx % 5), padding=2 ** (idx % 5))),
        nn.LeakyReLU()
      )
      for idx in range(depth)
    ])
    self.predict = spectral_norm(nn.Linear(128, 1))
    self.shape = shape
    self.down = down
    self.lookup = PositionLookup()

  def forward(self, distances, sequence):
    ind = torch.arange(distances.size(-1), device=distances.device)
    spatial = abs(ind[None, :].float() - ind[:, None].float())[None, None] / self.shape
    spatial = spatial.expand_as(distances)

    inputs = torch.cat((
      distances, spatial, sequence
    ), dim=1)
    inputs = (inputs + inputs.permute(0, 1, 3, 2)) / 2

    inputs[:, 0, ind, ind] = 0
    out = self.preprocess(inputs)
    count = 0
    for block in self.blocks:
      out = block(out) + out
      if count % self.down == 0:
        out = func.avg_pool2d(out, 2)
      count += 1
    out = func.adaptive_avg_pool2d(out, 1).view(-1, 128)
    out = self.predict(out)
    return out

class GaussianStupidEnergy(nn.Module):
  def __init__(self, depth=4, down=2, shape=64):
    super().__init__()
    self.preprocess = nn.Conv2d(1 + 42, 128, 5, padding=2)
    self.blocks = nn.ModuleList([
      nn.Sequential(
        nn.Conv2d(128, 128, 3, padding=1),
        nn.LeakyReLU(),
        nn.Conv2d(128, 128, 3, dilation=1, padding=1),
        nn.LeakyReLU()
      )
      for idx in range(depth)
    ])
    self.predict = nn.Linear(128, 1)
    self.shape = shape
    self.down = down
    self.lookup = PositionLookup()
    self.gaussian = nn.Parameter(torch.rand(1, 1, shape, shape, requires_grad=True))

  def forward(self, distances, sequence):
    inputs = torch.cat((
      distances, sequence
    ), dim=1)
    inputs = (inputs + inputs.permute(0, 1, 3, 2)) / 2
    ind = torch.arange(inputs.size(-1), device=inputs.device)
    inputs[:, 0, ind, ind] = 0
    gaussian_energy = ((inputs[:, 0] - self.gaussian) ** 2).reshape(inputs.size(0), 1, -1).mean(dim=-1)
    out = self.preprocess(inputs)
    count = 0
    for block in self.blocks:
      out = block(out) + out
      if count % self.down == 0:
        out = func.avg_pool2d(out, 2)
      count += 1
    out = func.adaptive_avg_pool2d(out, 1).view(-1, 128)
    out = self.predict(out)
    return out + gaussian_energy
