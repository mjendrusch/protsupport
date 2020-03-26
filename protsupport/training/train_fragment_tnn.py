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
from torchsupport.distributions import Mixture, VonMises
from torchsupport.structured import DataParallel as SDP
import torchsupport.structured as ts

from protsupport.modules.anglespace import PositionLookup
from protsupport.modules.protein_transformer import ProteinTransformer
from protsupport.data.proteinnet import ProteinNetKNN, ProteinNet

from protsupport.modules.losses import RGNLoss, StochasticRGNLoss, StochasticFullRGNLoss, WeightedAngleLoss
from protsupport.modules.backrub import Backrub
from protsupport.utils.geometry import orientation

def mixture_of_von_mises(weights, von_mises):
  distributions = []
  for params in von_mises:
    distributions.append(VonMises(*params))
  result = Mixture(distributions, weights)
  return result

class FragmentNet(ProteinNetKNN):
  def __init__(self, path, radius=8, num_neighbours=15, n_jobs=1, cache=True):
    ProteinNetKNN.__init__(
      self, path,
      num_neighbours=num_neighbours,
      n_jobs=n_jobs, cache=cache
    )
    self.radius = radius
    self.backrub = Backrub(n_moves=0)
    self.ors = torch.tensor(
      orientation(self.ters[1].numpy() / 100).transpose(2, 0, 1),
      dtype=torch.float
    )

  def autoregressive_structure(self, tertiary):
    pos = tertiary[:, 1]
    index_range = torch.arange(pos.size(0))
    distance = (pos[:, None] - pos[None, :]).norm(dim=-1)
    causal_distance = (index_range[:, None] - index_range[None, :])
    within_ball = torch.roll(distance < self.radius, 1, dims=0)
    within_ball[0] = False
    causal = causal_distance > 0
    admitted = causal * within_ball
    admitted_indices = admitted.nonzero()
    indices = admitted_indices[:, 0]
    connections = admitted_indices[:, 1]

    return ts.ScatterStructure(0, 0, indices, connections, node_count=pos.size(0))

  def get_mask(self, tertiary):
    mask = torch.rand(tertiary.size(0)) < torch.rand(1)[0]
    return mask

  def __getitem__(self, index):
    window = slice(self.index[index], self.index[index + 1])
    inds = self.inds[window]
    primary = self.pris[window] - 1

    # add noise:
    n_positions = random.randrange(max(1, primary.size(0) // 100))
    primary[torch.randint(0, primary.size(0), (n_positions,))] = torch.randint(0, 20, (n_positions,))

    tertiary = self.ters[:, :, window]
    distances, angles = self.backrub(tertiary[[0, 1, 3]].permute(2, 0, 1))
    angles = angles.permute(1, 0).contiguous()
    angles_gt = angles.clone()
    # angles = angles.roll(1, dims=0)
    # angles[0] = 0
    distances = distances / 100

    protein = SubgraphStructure(torch.zeros(distances.size(0), dtype=torch.long))
    neighbours = self.autoregressive_structure(distances)

    primary_onehot = torch.zeros(primary.size(0), 20, dtype=torch.float)
    primary_onehot[torch.arange(primary.size(0)), primary] = 1
    primary_onehot = primary_onehot.clamp(0, 1)

    assert neighbours.connections.max() < primary_onehot.size(0)
    inputs = (
      PackedTensor(angles),
      PackedTensor(distances),
      neighbours,
      protein
    )

    return inputs, PackedTensor(angles_gt, split=False)#, (PackedTensor(distances, split=False), PackedTensor(torch.ones(distances.size(0)), split=False), protein)

  def __len__(self):
    return ProteinNet.__len__(self)

class MixtureOfVonMisesLoss(nn.Module):
  def forward(self, inputs, target):
    dist = mixture_of_von_mises(*inputs)
    result = dist.log_prob(target)
    return -result.mean()

class AngleMSE(nn.Module):
  def forward(self, inputs, target):
    result = (1 - (inputs - target).cos()).mean()
    return result

def valid_callback(trn, inputs, outputs):
  with torch.no_grad():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    angles, angle_target = outputs[0]
    weights, parameters = angles
    mix = mixture_of_von_mises(weights, parameters)
    angles = mix.sample()
    means = torch.cat([param[0].unsqueeze(0) for param in parameters], dim=0)
    top = weights.argmax(dim=1)
    #angles = means[top, torch.arange(means.size(1))]
    ang = angles
    angles = angle_target

    # positions = positions.view(-1, 3)
    # ter = ter.view(-1, 3)

    # positions = positions[(struc.indices == 0).nonzero().view(-1)].numpy()
    # ter = ter[(struc.indices == 0).nonzero().view(-1)].numpy()

    # dst = np.linalg.norm((positions[None, :] - positions[:, None]), axis=-1)
    # dstt = np.linalg.norm((ter[None, :] - ter[:, None]), axis=-1)

    # ax.plot(positions[:, 0], positions[:, 1], positions[:, 2])
    # trn.writer.add_figure("output", fig, trn.step_id)

    # plt.close("all")

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')

    # ax.plot(ter[:, 0], ter[:, 1], ter[:, 2])
    # trn.writer.add_figure("input", fig, trn.step_id)
    # plt.close("all")

    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.imshow(dst)

    # trn.writer.add_figure("heat out", fig, trn.step_id)
    # plt.close("all")

    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.imshow(dstt)

    # trn.writer.add_figure("heat in", fig, trn.step_id)
    # plt.close("all")

    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.imshow(abs(dstt - dst))

    # trn.writer.add_figure("heat del", fig, trn.step_id)
    # plt.close("all")

    # fig = plt.figure()
    # ax = fig.add_subplot(111)

    # position_lookup = PositionLookup()
    # re_ter = position_lookup(angles[(struc.indices == 0).nonzero().view(-1)], struc.indices[(struc.indices == 0).nonzero().view(-1)])[0].view(-1, 3).numpy()

    # dsttt = np.linalg.norm((re_ter[None, :] - re_ter[:, None]), axis=-1)

    # ax.imshow(dsttt)
    # trn.writer.add_figure("heat expected", fig, trn.step_id)

    # plt.close("all")
    # trn.writer.add_scalar("size", float(ter.shape[0]), trn.step_id)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(angles[:, 1].numpy() % 6.3, angles[:, 2].numpy() % 6.3)
    ax.scatter(ang[:, 1].numpy() % 6.3, ang[:, 2].numpy() % 6.3)
    trn.writer.add_figure("rama", fig, trn.step_id)

    plt.close("all")

if __name__ == "__main__":
  with torch.autograd.detect_anomaly():
    data = FragmentNet(sys.argv[1])
    valid_data = FragmentNet(sys.argv[2])
    net = SDP(ProteinTransformer(6, 128, 10, heads=8, depth=3, neighbours=15, mix=20))
    training = SupervisedTraining(
      net, data, valid_data,
      [MixtureOfVonMisesLoss()],
      batch_size=16,
      max_epochs=1000,
      optimizer=lambda x: torch.optim.Adam(x, lr=1e-4, weight_decay=1e-4),
      device="cuda:0",
      network_name="aaattention/linear-dep",
      valid_callback=valid_callback
    ).load()
    final_net = training.train()
