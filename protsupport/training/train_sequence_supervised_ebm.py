import sys
import random

from matplotlib import pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as func
from torch.nn.utils.spectral_norm import spectral_norm
from torchsupport.training.energy import EnergyTraining
from torchsupport.training.energy_supervised import EnergySupervisedTraining
from torchsupport.training.samplers import (
  PackedMCMC, PackedDiscreteLangevin, PackedDiscreteGPLangevin,
  PackedHardDiscreteLangevin, IndependentSampler
)

from torchsupport.modules.basic import MLP
from torchsupport.data.io import to_device
from torchsupport.structured import PackedTensor, ConstantStructure, SubgraphStructure
from torchsupport.structured import DataParallel as SDP
from torchsupport.structured import scatter

from protsupport.data.proteinnet import ProteinNet, ProteinNetKNN
from protsupport.utils.geometry import orientation
from protsupport.modules.structures import RelativeStructure

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

    evolutionary = self.evos[:, window]
    tertiary = self.ters[:, :, window]
    orientation = self.ors[window, :, :].view(
      window.stop - window.start, -1
    )
    distances = self.ters[1, :, window].transpose(0, 1) / 100
    indices = torch.tensor(
      range(window.start, window.stop),
      dtype=torch.float
    )
    indices = indices.view(-1, 1)

    orientation = torch.cat((distances, orientation, indices), dim=1)
    angles = self.angs[:, window].transpose(0, 1)

    protein = SubgraphStructure(torch.zeros(indices.size(0), dtype=torch.long))
    neighbours = ConstantStructure(0, 0, (inds - self.index[index]).to(torch.long))

    sin = torch.sin(angles)
    cos = torch.cos(angles)
    angle_features = torch.cat((sin, cos), dim=1)

    primary_onehot = torch.zeros(primary.size(0), 20, dtype=torch.float)
    primary_onehot[torch.arange(primary.size(0)), primary] = 1
    #primary_onehot = primary_onehot + 0.05 * torch.randn_like(primary_onehot)
    primary_onehot = primary_onehot.clamp(0, 1)


    assert neighbours.connections.max() < primary_onehot.size(0)
    inputs = (
      PackedTensor(primary_onehot),
      PackedTensor(primary_onehot), # gt_ignore
      PackedTensor(angle_features),
      PackedTensor(orientation),
      neighbours,
      protein
    )

    return inputs, PackedTensor(primary, split=False)

  def __len__(self):
    return ProteinNet.__len__(self)

class SequenceEBM(nn.Module):
  def __init__(self, in_size, out_size=20,
               hidden_size=128, neighbours=30,
               w_depth=3, p_depth=3,
               max_distance=20, distance_kernels=16):
    super().__init__()
    self.features = MLP(
      2 * in_size, hidden_size,
      hidden_size=hidden_size,
      depth=p_depth,
      batch_norm=False,
      normalization=spectral_norm
    )
    self.weight = MLP(
      2 * in_size, 1,
      hidden_size=hidden_size,
      depth=w_depth,
      batch_norm=False,
      normalization=spectral_norm
    )
    self.out = MLP(
      hidden_size * neighbours, out_size,
      hidden_size=[300, hidden_size, hidden_size],
      batch_norm=False,
      normalization=spectral_norm
    )
    self.rbf = (0, max_distance, distance_kernels)

  def forward(self, primary, gt_ignore, angles, orientation, neighbours, protein,
              return_deltas=False):
    assert neighbours.connections.max() < primary.size(0)
    indices = neighbours.connections
    sequence = primary.clone()
    primary = primary[indices].clone()
    primary[:, 0] = 0
    angles = angles[indices]
    relative = RelativeStructure(neighbours, self.rbf)
    orientation = relative.message(
      orientation, orientation
    )
    features = torch.cat((
      primary, angles, orientation
    ), dim=2).permute(0, 2, 1)

    inputs = torch.cat((
      features, features[:, :, 0:1].expand_as(features)
    ), dim=1)
    in_view = inputs.transpose(2, 1).reshape(
      -1, 2 * features.size(1)
    )
    p = self.features(in_view)
    w = self.weight(in_view)
    prod = (p * w)
    cat = prod.reshape(inputs.size(0), -1)
    out = self.out(cat)
    this_energy = out[torch.arange(out.size(0)), sequence.argmax(dim=1)]
    differences = this_energy.unsqueeze(-1) - out
    energy = scatter.mean(this_energy.unsqueeze(-1), protein.indices)
    if return_deltas:
      return (energy, -out), out
    return energy, out

class EBMTraining(EnergySupervisedTraining):
  def classifier_loss(self, logits, labels):
    _, logits = logits
    return func.cross_entropy(-logits, labels)

  def logit_energy(self, logits, *args):
    logits, _ = logits
    return logits

  def prepare(self):
    index = random.randrange(0, len(self.data))
    (primary, ground_truth, angles, orientation, neighbours, protein), _ = self.data[index]
    primary_indices = torch.randint(0, 20, (primary.tensor.size(0),))
    primary = torch.zeros_like(primary.tensor)
    primary[torch.arange(primary.size(0)), primary_indices] = 1
    return (
      PackedTensor(primary), ground_truth,
      angles, orientation, neighbours, protein
    )

  def decompose_batch(self, data, *args):
    count = len(data)
    targets = [self.device] * count
    gt, angles, orientation, neighbours, protein = args
    neighbours = neighbours.chunk(targets)
    protein = protein.chunk(targets)
    result = [
      to_device((
        data[idx].detach(),
        gt[idx].detach(),
        angles[idx].detach(),
        orientation[idx].detach(),
        neighbours[idx],
        protein[idx]
      ), "cpu")
      for idx in range(count)
    ]

    return result

  def each_generate(self, data, ground_truth, *args):
    datamax = data.tensor.argmax(dim=1)
    gtmax = ground_truth.tensor.argmax(dim=1)

    identity = (datamax == gtmax).float().mean()
    self.writer.add_scalar("identity", float(identity), self.step_id)

    datamax = data[0].tensor.argmax(dim=1)
    gtmax = ground_truth[0].tensor.argmax(dim=1)

    dataseq = "".join([AA_CODE[int(aa)] for aa in datamax])
    gtseq = "".join([AA_CODE[int(aa)] for aa in gtmax])
    self.writer.add_text("dataseq", dataseq, self.step_id)
    self.writer.add_text("gtseq", gtseq, self.step_id)

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
  net = SDP(SequenceEBM(51, 20, neighbours=15))
  integrator = IndependentSampler(steps=20, scale=10, rate=1, noise=0.05)
  training = EBMTraining(
    net, data,
    batch_size=32,
    max_epochs=1000,
    integrator=integrator,
    buffer_probability=0.95,
    decay=1.0,
    buffer_size=10000,
    optimizer_kwargs={"lr": 1e-4},
    device="cuda:0",
    network_name="sequence-ebm/independent-sampler-26",
    verbose=True
  )
  final_net = training.train()
