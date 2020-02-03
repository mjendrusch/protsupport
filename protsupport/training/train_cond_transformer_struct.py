import sys
import random
import numpy as np

from matplotlib import pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as func
from torchsupport.training.training import SupervisedTraining

from torchsupport.structured import PackedTensor, ConstantStructure
from torchsupport.structured import DataParallel as SDP
from protsupport.training.train_sequence_ebm import DDP

from protsupport.data.proteinnet import ProteinNet, ProteinNetKNN
from protsupport.utils.geometry import orientation
from protsupport.modules.cond_structured_transformer import ConditionalStructuredTransformer
from protsupport.modules.backrub import Backrub

def valid_callback(training, data, predictions):
  inputs, labels = data
  confusion = torch.zeros(20, 20)
  for label, prediction in zip(labels[0], predictions[0][0]):
    pred = prediction.argmax(dim=0)
    confusion[label, pred] += 1
  fig, ax = plt.subplots()
  ax.imshow(confusion / confusion.sum(dim=1, keepdim=True), cmap="Reds")
  training.writer.add_figure("confusion", fig, training.step_id)
  # for name, parameter in training.net.named_parameters():
  #   training.writer.add_histogram(f"phist {name}", parameter.detach().cpu().numpy(), training.step_id)

class CondTransformerNet(ProteinNetKNN):
  def __init__(self, path, num_neighbours=20, n_jobs=1, n_backrub=10,
               phi=0.2 * np.pi, psi=0.2 * np.pi, tau=0.2 * np.pi, cache=True):
    super(CondTransformerNet, self).__init__(
      path,
      num_neighbours=num_neighbours,
      n_jobs=n_jobs, cache=cache
    )
    self.backrub = Backrub(n_moves=n_backrub, phi=phi, psi=psi, tau=tau)
    self.ors = torch.tensor(
      orientation(self.ters[1].numpy() / 100).transpose(2, 0, 1),
      dtype=torch.float
    )

  def __getitem__(self, index):
    # Extract the boundaries of a whole protein
    window = slice(self.index[index], self.index[index + 1])
    seq_len = window.stop - window.start

    # Make me a mask
    mask = np.random.choice(seq_len, size=np.random.randint(np.ceil(0.05*seq_len), seq_len), replace=True)
    mask = torch.tensor(mask, dtype=torch.long)
    mask_binary = torch.zeros(seq_len, dtype=torch.uint8)
    mask_binary[mask] = 1

    # Get sequence info
    primary = self.pris[window] - 1

    primary_masked = primary.clone()
    primary_masked[mask] = 20
    primary_onehot = torch.zeros((seq_len, 21), dtype=torch.float)
    primary_onehot[torch.arange(seq_len), primary_masked] = 1

    # Prepare orientation infos
    orientation = self.ors[window, :, :].view(seq_len, -1)

    tertiary = self.ters[:, :, window]
    distances, angles = self.backrub(tertiary[[0, 1, 3]].permute(2, 0, 1))
    distances = distances[:, 1] / 100
    angles = angles.transpose(0, 1)

    indices = torch.tensor(
      range(window.start, window.stop),
      dtype=torch.float
    )
    indices = indices.view(-1, 1)
    orientation = torch.cat((distances, orientation, indices), dim=1)

    # Prepare neighborhood structure
    inds = self.inds[window]
    neighbours = ConstantStructure(0, 0, (inds - self.index[index]).to(torch.long))

    # Prepare angle features
    sin = torch.sin(angles)
    cos = torch.cos(angles)
    angle_features = torch.cat((sin, cos), dim=1)

    inputs = (
      PackedTensor(angle_features),
      PackedTensor(primary_onehot),
      PackedTensor(orientation),
      neighbours
    )

    targets = (
      PackedTensor(primary, split=False),
      PackedTensor(mask_binary, split=False)
      )

    return inputs, targets

  def __len__(self):
    return ProteinNet.__len__(self)

class MaskedLoss(nn.Module):
  def __init__(self):
    super().__init__()
    self.loss = nn.CrossEntropyLoss()

  def forward(self, inputs, targets):
    targ, mask = targets
    return self.loss(inputs[mask], targ[mask])
    # except:
    #   return torch.tensor(0, dtype=inputs.dtype, device=inputs.device,)

class ConditionalStructuredTransformerTraining(SupervisedTraining):
  """
  Train a transformer model mapping structure to sequence for each position as a conditional probability of its surroundings
  """
  def each_step(self):
    super().each_step()
    # Schedule from 'Attention is all you need'
    learning_rate = torch.pow(torch.tensor(128.0), -0.5)
    step_num = torch.tensor(float(self.step_id + 1))
    learning_rate *= min(
      torch.pow(step_num, -0.5),
      step_num * torch.pow(torch.tensor(4000.0), -1.5)
    )
    self.optimizer.param_groups[0]["lr"] = learning_rate

if __name__ == "__main__":
  data = CondTransformerNet(sys.argv[1], num_neighbours=15)
  valid_data = CondTransformerNet(sys.argv[2], num_neighbours=15)
  net = SDP(
    ConditionalStructuredTransformer(
    6, 128, 10, 
    attention_size=128, heads=8,
    mlp_depth=2, depth=6, batch_norm=True
  ))
  training = ConditionalStructuredTransformerTraining(
    net, data, valid_data,
    [MaskedLoss()],
    batch_size=8,
    max_epochs=1000,
    optimizer=lambda x: torch.optim.Adam(x), # LR scheduled 
    device="cuda:0",
    network_name="cond-structured-transformer/15-6-drop-10-rub",
    valid_callback=valid_callback,
    report_interval=10
  ).load()
  final_net = training.train()
