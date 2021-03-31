import sys
import random

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import torch
import torch.nn as nn
import torch.nn.functional as func
from torch.nn.utils.spectral_norm import spectral_norm
from torchsupport.training.energy import EnergyTraining, DenoisingScoreTraining, SlicedScoreTraining
from torchsupport.training.samplers import AnnealedPackedLangevin
from torchsupport.optim.diffmod import DiffMod

from torchsupport.modules.basic import MLP
from torchsupport.data.io import to_device, make_differentiable
from torchsupport.structured import PackedTensor, ConstantStructure, SubgraphStructure
from torchsupport.structured import DataParallel as SDP
from torchsupport.structured import scatter

from protsupport.data.proteinnet import ProteinNet, ProteinNetKNN
from protsupport.utils.geometry import orientation
from protsupport.modules.structures import RelativeStructure
from protsupport.modules.structured_score import StructuredScore
from protsupport.modules.anglespace import PositionLookup
from protsupport.modules.backrub import Backrub
from protsupport.modules.transformer import attention_connected, linear_connected, assignment_connected

AA_CODE = "ACDEFGHIKLMNPQRSTVWY"

class EBMNet(ProteinNetKNN):
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

    # add noise:
    n_positions = random.randrange(max(1, primary.size(0) // 100))
    primary[torch.randint(0, primary.size(0), (n_positions,))] = torch.randint(0, 20, (n_positions,))

    tertiary = self.ters[:, :, window]
    distances, angles = self.backrub(tertiary[[0, 1, 3]].permute(2, 0, 1))
    distances = distances[:, 1] / 100

    protein = SubgraphStructure(torch.zeros(distances.size(0), dtype=torch.long))
    neighbours = ConstantStructure(0, 0, (inds - self.index[index]).to(torch.long))

    primary_onehot = torch.zeros(primary.size(0), 20, dtype=torch.float)
    primary_onehot[torch.arange(primary.size(0)), primary] = 1
    primary_onehot = primary_onehot.clamp(0, 1)

    assert neighbours.connections.max() < primary_onehot.size(0)
    inputs = (
      PackedTensor((angles + 0.1 * torch.randn_like(angles)).permute(1, 0).contiguous()),
      PackedTensor(primary_onehot.contiguous()),
      protein
    )

    return inputs

  def __len__(self):
    return ProteinNet.__len__(self)

class EBMTraining(SlicedScoreTraining):
  def prepare(self):
    index = random.randrange(0, len(self.data))
    (positions, ground_truth, protein) = self.data[index]
    angles = 3.14 * torch.randn_like(positions.tensor)
    return (
      PackedTensor(angles), ground_truth, protein
    )

  def sample(self):
    self.score.eval()
    with torch.no_grad():
      integrator = AnnealedPackedLangevin([
        self.sigma * self.factor ** idx for idx in range(self.n_sigma)
      ])
      prep = to_device(self.prepare_sample(), self.device)
      data, *args = self.data_key(prep)
      result = integrator.integrate(self.score, data, *args).detach()
    self.score.train()
    return to_device((result, data, *args), self.device)

  def noise(self, data):
    scale = torch.randint(0, self.n_sigma, (len(data.lengths),))
    scale = torch.repeat_interleave(scale, torch.tensor(data.lengths, dtype=torch.long))
    sigma = self.sigma * self.factor ** scale.float()
    sigma = sigma.to(self.device)
    sigma = sigma.view(*sigma.shape, *((data.tensor.dim() - sigma.dim()) * [1]))
    noise = data.clone()
    noise.tensor = data.tensor + sigma * torch.randn_like(data.tensor)
    tmp = data.clone()
    tmp.tensor = sigma
    sigma = tmp
    print("CHECK", data.lengths, sigma.lengths, data.tensor.shape, sigma.tensor.shape)
    return noise, sigma

  def energy_loss(self, score, data, noisy, sigma):
    sigma = sigma.tensor
    vectors = self.noise_vectors(score)
    make_differentiable(vectors)

    grad_v = (score * vectors).view(score.size(0), -1).sum()
    jacobian = torch.autograd.grad(grad_v, noisy.tensor, retain_graph=True, create_graph=True)[0]
    # jacograd = torch.autograd.grad(jacobian.mean(), noisy.tensor, retain_graph=True)
    # print(jacobian)
    # print(jacograd)

    norm = (score ** 2).view(score.size(0), -1).sum(dim=-1) / 2
    jacobian = (vectors * jacobian).view(score.size(0), -1).sum(dim=-1)

    result = (norm + jacobian) * sigma.view(score.size(0), -1) ** 2

    result = result.mean()

    self.current_losses["ebm"] = float(result)

    return result

  # def energy_loss(self, score, data, noisy, sigma):
  #   raw_loss = 0.5 * sigma ** 2 * ((score + (noisy.tensor - data.tensor) / sigma ** 2) ** 2)

  #   raw_loss = raw_loss.sum(dim=1, keepdim=True)
  #   self.current_losses["ebm"] = float(raw_loss.mean())
  #   return raw_loss.mean()

  def each_generate(self, data, noise, gt, protein):
    with torch.no_grad():
      lookup = PositionLookup()
      angs = data.tensor.cpu()
      c_alpha, _ = lookup(angs[protein.indices.cpu() == 0], torch.zeros_like(protein.indices[protein.indices == 0].cpu()))
      c_alpha = c_alpha[:, 1]
      dist = (c_alpha[:, None] - c_alpha[None, :]).norm(dim=-1)
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

  def forward(self, *args):
    inputs = []
    for arg in args:
      if isinstance(arg, PackedTensor):
        inputs.append(arg.tensor)
      else:
        inputs.append(arg)
    return self.net(*inputs)

torch.backends.cudnn.enabled = False

if __name__ == "__main__":
  data = EBMNet(sys.argv[1], num_neighbours=15)
  net = SDP(
    StructuredScore(
      6, 128, 10,
      attention_size=32, heads=8,
      mlp_depth=2, depth=3, batch_norm=True, dropout=0.1,
      neighbours=15, angles=True, distance_kernels=32,
      connected=attention_connected
    )
  )
  training = EBMTraining(
    net, data,
    sigma=3.15,
    batch_size=32,
    max_epochs=1000,
    #optimizer=DiffMod,
    optimizer_kwargs={"lr": 1e-4},
    device="cuda:0",
    network_name="distance-gan/pooled-score",
    verbose=True,
    report_interval=100
  ).load()
  final_net = training.train()
