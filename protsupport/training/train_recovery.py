import sys
import random

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from scipy.spatial import cKDTree

import torch
import torch.nn as nn
import torch.nn.functional as func
from torchsupport.training.training import SupervisedTraining
from torchsupport.structured import PackedTensor, SubgraphStructure, scatter, ConstantStructure
from torchsupport.modules.basic import one_hot_encode
from torchsupport.structured import DataParallel as SDP

from protsupport.modules.anglespace import PositionLookup
from protsupport.modules.recovery import Recovery
from protsupport.data.proteinnet import ProteinNetKNN, ProteinNet

from protsupport.modules.losses import RGNLoss, StochasticRGNLoss, StochasticFullRGNLoss, WeightedAngleLoss
from protsupport.modules.backrub import Backrub
from protsupport.utils.geometry import orientation

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
    window = slice(self.index[index], self.index[index + 1])
    inds = self.inds[window]
    primary = self.pris[window] - 1

    if primary.size(0) < 30:
      return GANNet.__getitem__(self, (index + 1) % len(self))

    # add noise:
    n_positions = random.randrange(max(1, primary.size(0) // 100))
    primary[torch.randint(0, primary.size(0), (n_positions,))] = torch.randint(0, 20, (n_positions,))

    tertiary = self.ters[:, :, window]
    distances, angles = self.backrub(tertiary[[0, 1, 3]].permute(2, 0, 1))
    angles = angles.permute(1, 0).contiguous()
    distances = distances / 100

    protein = SubgraphStructure(torch.zeros(distances.size(0), dtype=torch.long))
    neighbours = ConstantStructure(0, 0, (inds - self.index[index]).to(torch.long))

    primary_onehot = torch.zeros(primary.size(0), 20, dtype=torch.float)
    primary_onehot[torch.arange(primary.size(0)), primary] = 1
    primary_onehot = primary_onehot.clamp(0, 1)

    assert neighbours.connections.max() < primary_onehot.size(0)
    inputs = (
      PackedTensor(angles),
      protein
    )

    return inputs, PackedTensor(angles, split=False), (PackedTensor(distances, split=False), PackedTensor(torch.ones(distances.size(0)), split=False), protein)

  def __len__(self):
    return ProteinNet.__len__(self)

class AngleMSE(nn.Module):
  def forward(self, inputs, target):
    #result = (1 - (inputs - target).cos()).mean()
    result = (inputs.sin() - target.sin()) ** 2
    result += (inputs.cos() - target.cos()) ** 2
    return result.sum(dim=1).mean()

def valid_callback(trn, inputs, outputs):
  with torch.no_grad():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    positions, pos_target = outputs[1]
    ter, mask, struc = pos_target
    angles, angle_target = outputs[0]
    ang = angles
    angles = angle_target

    positions = positions.view(-1, 3)
    ter = ter.view(-1, 3)

    positions = positions[(struc.indices == 0).nonzero().view(-1)].numpy()
    ter = ter[(struc.indices == 0).nonzero().view(-1)].numpy()

    dst = np.linalg.norm((positions[None, :] - positions[:, None]), axis=-1)
    dstt = np.linalg.norm((ter[None, :] - ter[:, None]), axis=-1)

    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2])
    trn.writer.add_figure("output", fig, trn.step_id)

    plt.close("all")

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(ter[:, 0], ter[:, 1], ter[:, 2])
    trn.writer.add_figure("input", fig, trn.step_id)
    plt.close("all")

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(dst)

    trn.writer.add_figure("heat out", fig, trn.step_id)
    plt.close("all")

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(dstt)

    trn.writer.add_figure("heat in", fig, trn.step_id)
    plt.close("all")

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(abs(dstt - dst))

    trn.writer.add_figure("heat del", fig, trn.step_id)
    plt.close("all")

    fig = plt.figure()
    ax = fig.add_subplot(111)

    position_lookup = PositionLookup()
    re_ter = position_lookup(angles[(struc.indices == 0).nonzero().view(-1)], struc.indices[(struc.indices == 0).nonzero().view(-1)])[0].view(-1, 3).numpy()

    dsttt = np.linalg.norm((re_ter[None, :] - re_ter[:, None]), axis=-1)

    ax.imshow(dsttt)
    trn.writer.add_figure("heat expected", fig, trn.step_id)

    plt.close("all")
    trn.writer.add_scalar("size", float(ter.shape[0]), trn.step_id)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(angles[:, 1].numpy() % 6.3, angles[:, 2].numpy() % 6.3)
    ax.scatter(ang[:, 1].numpy() % 6.3, ang[:, 2].numpy() % 6.3)
    trn.writer.add_figure("rama", fig, trn.step_id)

    plt.close("all")

if __name__ == "__main__":
  data = GANNet(sys.argv[1])
  valid_data = GANNet(sys.argv[2])
  net = SDP(Recovery(128, 10, heads=8, depth=3, neighbours=15))
  training = SupervisedTraining(
    net, data, valid_data,
    [AngleMSE(), StochasticRGNLoss(1000, relative=True)],
    batch_size=8,
    max_epochs=1000,
    optimizer=lambda x: torch.optim.Adam(x, lr=1e-4),
    device="cuda:0",
    network_name="recovery",
    valid_callback=valid_callback
  ).load()
  final_net = training.train()
