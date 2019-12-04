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
from torchsupport.structured import PackedTensor, SubgraphStructure, scatter
from torchsupport.modules.basic import one_hot_encode
from torchsupport.structured import DataParallel as SDP

from protsupport.modules.anglespace import PositionLookup
from protsupport.modules.rgn import ConvolutionalGN, ResidualGN, RGN, TransformerGN
from protsupport.data.proteinnet import ProteinNet

from protsupport.modules.losses import StochasticRGNLoss, WeightedAngleLoss

class RGNNet(ProteinNet):
  def __init__(self, path):
    ProteinNet.__init__(self, path)

  def __getitem__(self, index):
    result = super().__getitem__(index)
    primary = result["primary"][:500]
    evolutionary = result["evolutionary"][:, :500].t()
    tertiary = result["tertiary"] / 100
    tertiary = tertiary[[0, 1, 3], :, :].permute(2, 0, 1).contiguous()[:500].view(-1, 3)
    angles = result["angles"].contiguous().view(-1, 3)[:500].contiguous()
    mask = result["mask"][:500].view(-1)

    print(angles.min(), angles.max())

    mask = mask#torch.repeat_interleave(mask, 3)

    membership = SubgraphStructure(torch.zeros(primary.size(0), dtype=torch.long))
    primary_onehot = one_hot_encode(primary - 1, range(20)).t()
    primary_onehot = torch.cat((primary_onehot, evolutionary), dim=1)

    inputs = (
      PackedTensor(primary_onehot),
      membership
    )

    outputs = (
      PackedTensor(tertiary, split=False), PackedTensor(mask.unsqueeze(1), split=False), membership
    )

    print("tsize", angles.size())

    return inputs, outputs, (PackedTensor(angles, split=False), PackedTensor(mask.view(-1), split=False))

  def __len__(self):
    return ProteinNet.__len__(self)

# class StructuredMaskedCE(nn.Module):
#   def __init__(self):
#     super().__init__()
#     #self.loss = nn.MSELoss(reduction="sum")

#   def forward(self, inputs, target):
#     target, mask, structure = target
#     target = target.view(-1, 3, 3)[:, 1]#target.view(-1, 3)
#     inputs = inputs.view(-1, 3, 3)[:, 1]#inputs.view(-1, 3)
#     dst = lambda x, y: (x - y).norm(dim=1)
#     msk = lambda x, y: (x * y).squeeze(dim=1)
#     indices = structure.indices#torch.repeat_interleave(structure.indices, 3)
#     indices = indices[(mask > 0).nonzero().view(-1)]
#     target = target[(mask > 0).nonzero().view(-1)]
#     inputs = inputs[(mask > 0).nonzero().view(-1)]
#     # unique, counts = indices.unique(return_counts=True)
#     print("idt", indices.dtype)
#     # mask, _ = scatter.pairwise_no_pad(msk, mask, indices)
#     target, rmsd_indices = scatter.pairwise_no_pad(dst, target, indices)
#     inputs, _ = scatter.pairwise_no_pad(dst, inputs, indices)
#     print(target.shape, mask.shape, inputs.shape)
#     print("rdt", rmsd_indices.dtype)
#     result = (100 * inputs - 100 * target) ** 2
#     unique, counts = structure.indices.unique(return_counts=True)
#     denominator = (counts * (counts - 1)).float().to(result.device)
#     result = torch.sqrt(2 * scatter.add(result, rmsd_indices.to(result.device)) + 1e-6) / torch.sqrt(denominator) / counts.float().to(result.device)
#     return result.mean()#torch.sqrt(result.mean() + 1e-6)

class AngleMSE(nn.Module):
  def forward(self, inputs, target):
    target, mask = target
    mask = mask[:inputs.size(0)]
    target = target[mask.nonzero().view(-1)]
    inputs = inputs[mask.nonzero().view(-1)]
    #print("ITS", inputs.shape, target.shape)
    inputs = inputs.view(-1)
    target = target.view(-1)[:inputs.size(0)]
    #print("ITS0", inputs.shape, target.shape, mask.shape)
    #return (((inputs - inputs) % (2 * np.pi)) ** 2).mean() / 10
    result = ((inputs.sin() - target.sin()) ** 2).mean() + ((inputs.cos() - target.cos()) ** 2).mean()
    result = result / 10.0
    return result

def valid_callback(trn, inputs, outputs):
  with torch.no_grad():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    positions, pos_target = outputs[0]
    ter, mask, struc = pos_target
    angles, angle_target = outputs[1]
    angles, _ = angle_target

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

if __name__ == "__main__":
  data = RGNNet(sys.argv[1])
  valid_data = RGNNet(sys.argv[2])
  net = SDP(RGN(41))
  training = SupervisedTraining(
    net, data, valid_data,
    [StochasticRGNLoss(100, relative=True), AngleMSE()],
    batch_size=64,
    max_epochs=1000,
    optimizer=lambda x: torch.optim.Adam(x, lr=5e-5),
    device="cuda:0",
    network_name="rgn-test/stochastic-5-e-5-fixactivation",
    valid_callback=valid_callback
  )
  final_net = training.train()
