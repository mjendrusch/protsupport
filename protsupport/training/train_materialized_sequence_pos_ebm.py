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
from torchsupport.training.samplers import PackedLangevin, AugmentedLangevin

from torchsupport.modules.basic import MLP
from torchsupport.modules import replace_gradient
from torchsupport.structured.modules.materialized_transformer import (
  MaterializedTransformerBlock, MaterializedMultiHeadAttention
)
from torchsupport.data.io import to_device
from torchsupport.structured import PackedTensor, ConstantStructure, SubgraphStructure
from torchsupport.structured import DataParallel as SDP
from torchsupport.structured import scatter

from protsupport.data.proteinnet import ProteinNet, ProteinNetKNN
from protsupport.utils.geometry import orientation
from protsupport.modules.structures import RelativeStructure
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
    start = self.index[index]
    end = self.index[index + 1]
    rstart = start
    if end - start > N:
      rstart = random.randrange(start, end - N)
    rend = rstart + N
    window = slice(rstart, rend)
    inds = self.inds[window]
    primary = self.pris[window] - 1

    # get sequence positions
    keeps = self.keeps[window]
    keeps = keeps - keeps[0]

    # add noise:
    n_positions = random.randrange(max(1, primary.size(0) // 5))
    primary[torch.randint(0, primary.size(0), (n_positions,))] = torch.randint(0, 20, (n_positions,))

    tertiary = self.ters[:, :, window]
    distances, angles = self.backrub(tertiary[[0, 1, 3]].permute(2, 0, 1))
    distances = tertiary.permute(2, 0, 1)[:, 1] / 100

    # distances = (distances[None, :] - distances[:, None]).norm(dim=-1).unsqueeze(0) / 40
    # distances = 0.99 * distances + 0.01 * torch.rand_like(distances)
    # distances = distances.clamp(0, 1)
    # print(distances.max())
    sequence = torch.zeros(42, N, N)
    sequence[-1] = 1

    protein = SubgraphStructure(torch.zeros(distances.size(0), dtype=torch.long))
    neighbours = ConstantStructure(0, 0, (inds - self.index[index]).to(torch.long))

    primary_onehot = torch.zeros(primary.size(0), 20, dtype=torch.float)
    primary_onehot[torch.arange(primary.size(0)), primary] = 1
    primary_onehot = primary_onehot + 0.1 * torch.rand_like(primary_onehot)
    primary_onehot = primary_onehot.clamp(0, 1)

    inputs = (
      (distances, primary_onehot.transpose(0, 1)),
      keeps
    )

    return inputs

  def __len__(self):
    return len(self.valid_indices)

class EBMTraining(EnergyTraining):
  def prepare(self):
    index = random.randrange(0, len(self.data))
    (distance, ground_truth), keep = self.data[index]
    maxval = keep.max()
    minval = keep.min()
    mval = maxval - minval
    perm = torch.randperm(mval + 1)
    keep = perm[:ground_truth.size(-1)]
    keep = keep.sort().values
    keep = keep - keep[0]
    distance = 1 * torch.randn_like(distance)
    ground_truth = torch.rand_like(ground_truth)
    ind = torch.arange(ground_truth.size(1))
    amx = ground_truth.argmax(dim=0)
    ground_truth[amx, ind] = 1
    return (
      (distance, ground_truth),
      keep
    )

  def each_generate(self, data, indices):
    with torch.no_grad():
      dist = data[0].cpu()
      seq = data[1].cpu()
      tmp = torch.zeros_like(seq)
      ind_b = torch.arange(tmp.size(0), dtype=torch.long, device=tmp.device)
      ind_s = torch.arange(tmp.size(2), dtype=torch.long, device=tmp.device)
      amx = seq.argmax(dim=1)
      tmp[ind_b[:, None, None], amx[:, None, :], ind_s[None, None, :]] = 1
      seq = tmp[:, None].repeat_interleave(3, dim=1)
      dist = (dist[:, None, :] - dist[:, :, None]).norm(dim=-1)[:, None] / 100
      dist = dist
      dist = (dist + dist.permute(0, 1, 3, 2)) / 2
      dist[:, :, torch.arange(dist.size(-1)), torch.arange(dist.size(-1))] = 0
      dist = (dist - dist.min()) / (dist.max() - dist.min())
      dist = dist.repeat_interleave(3, dim=1).numpy()
      self.writer.add_images("dist", dist, self.step_id)
      self.writer.add_image("seq", seq[0], self.step_id)

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

class MaterializedEnergy(nn.Module):
  def __init__(self,
               pair_out_sizes=None,
               pair_proj_sizes=None,
               kernel_size=1, heads=8,
               drop=None, seq_depth=10,
               attention_size=64,
               value_size=None,
               pair_depth=100, size=64,
               full=False, similarity=None):
    super().__init__()
    self.project = nn.Conv2d(1 + 100, size, 1, padding=0)
    self.blocks = nn.ModuleList([
      MaterializedTransformerBlock(
        size, size, size, size,
        attention_size=attention_size, heads=heads,
        value_size=(value_size or size), kernel_size=kernel_size,
        activation=nn.LeakyReLU(0.2), dropout=drop or 0.1,
        full=full, similarity=similarity
      )
      for idx in range(pair_depth)
    ])
    self.dist_embedding = nn.Conv2d(1, 100, 1)
    self.edge_project = nn.Conv2d(size + 100, size, 1, bias=True)
    self.node_project = nn.Conv1d(2 * size + 20 + 100, size, 1, bias=True)
    self.decision = nn.Conv1d(size, 1, 1, bias=True)

  def position_embedding(self, pos):
    pos = abs(pos[:, None, :] - pos[:, :, None])
    ind = torch.arange(50, dtype=torch.float)[None, :, None, None]
    ind = ind.to(pos.device)
    features = pos[:, None].repeat_interleave(50, dim=1)
    features = features / (10000 ** (2 * ind / 100))
    sin = features.sin()
    cos = features.cos()
    features = torch.cat((sin, cos), dim=1)
    return features

  def seq_position_embedding(self, pos):
    ind = torch.arange(50, dtype=torch.float)[None, :, None]
    ind = ind.to(pos.device)
    features = pos[:, None].repeat_interleave(50, dim=1)
    features = features / (10000 ** (2 * ind / 100))
    sin = features.sin()
    cos = features.cos()
    features = torch.cat((sin, cos), dim=1)
    return features

  def distance_embedding(self, distances):
    ind = torch.arange(50, dtype=torch.float)
    ind = ind[None, :, None, None]
    distances = distances / (10000 ** (2 * ind / 100))
    sin = distances.sin()
    cos = distances.cos()
    return torch.cat((sin, cos), dim=1)

  def tile(self, data, indices):
    data = (data + data.permute(0, 1, 3, 2)) / 2
    projected = self.project(data)
    seq = torch.cat((
      projected.mean(dim=-1),
      projected.std(dim=-1)
    ), dim=1)
    pos = self.position_embedding(indices)
    result = torch.cat((projected, pos), dim=1)
    pos = self.seq_position_embedding(indices)
    seq = torch.cat((seq, pos), dim=1)
    return seq, result

  def predict(self, nodes, edges):
    out = self.decision(func.relu(nodes)).mean(dim=-1)
    return out

  def forward(self, data, indices):
    inputs, sequence = data
    tmp = torch.zeros_like(sequence)
    ind_b = torch.arange(tmp.size(0), dtype=torch.long, device=tmp.device)
    ind_s = torch.arange(tmp.size(2), dtype=torch.long, device=tmp.device)
    amx = sequence.argmax(dim=1)
    tmp[ind_b[:, None, None], amx[:, None, :], ind_s[None, None, :]] = 1
    sequence = replace_gradient(tmp, sequence)
    inputs = (inputs[:, None, :] - inputs[:, :, None]).norm(dim=-1)[:, None] / 100
    inputs = torch.cat((inputs, (self.dist_embedding(inputs))), dim=1)
    nodes, edges = self.tile(inputs, indices)
    edges = self.edge_project(edges)
    nodes = torch.cat((nodes, sequence), dim=1)
    nodes = self.node_project(nodes)
    mask = torch.ones(nodes.size(0), nodes.size(-1), dtype=torch.bool, device=nodes.device)

    for block in self.blocks:
      nodes, edges = block(nodes, edges, mask)
    predictions = self.predict(nodes, edges)

    return predictions

def transform(data):
  struc, seq = data
  flip = random.random() < 0.5
  offset = 1.0 * torch.randn_like(struc)
  offset = offset.cumsum(dim=-1)
  struc = struc + offset - struc.mean(dim=-1, keepdim=True)
  if flip:
    count = random.randrange(1, 10)
    seq = seq.roll(count, dims=2)
  else:
    seq = (seq + torch.rand_like(seq)).clamp(0, 1)
  return (struc, seq)

if __name__ == "__main__":
  data = EBMNet(sys.argv[1], num_neighbours=15, N=64)
  net = SDP(
    MaterializedEnergy(
      pair_depth=4, size=128, value_size=16,
      heads=8, kernel_size=1, drop=0.1, full=True
    )
  )
  integrator = AugmentedLangevin(
    rate=(5000.0, 50.0),
    noise=(0.1, 0.1),
    steps=50,
    transform_interval=50,
    transform=transform,
    max_norm=None,
    clamp=(None, (0, 1))
  )
  training = EBMTraining(
    net, data,
    batch_size=8,
    decay=1.0,
    max_epochs=5000,
    integrator=integrator,
    buffer_probability=0.999,
    buffer_size=10000,
    optimizer_kwargs={"lr": 5e-4, "betas": (0.0, 0.99)},
    device="cuda:0",
    network_name="materialized-ebm-new/no-reinit-5",
    verbose=True,
    checkpoint_interval=50,
    report_interval=1
  ).load()
  final_net = training.train()

