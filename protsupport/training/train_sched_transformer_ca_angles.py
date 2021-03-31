import sys
import random
import numpy as np

from matplotlib import pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as func
from torchsupport.training.training import SupervisedTraining

from torchsupport.structured import PackedTensor, ConstantStructure, SubgraphStructure
from torchsupport.structured import DataParallel as SDP
from protsupport.training.train_sequence_ebm import DDP

from protsupport.data.proteinnet import ProteinNet, ProteinNetKNN
from protsupport.utils.geometry import orientation
from protsupport.modules.scheduled_structured_transformer import ScheduledPrestructuredTransformer
from protsupport.modules.structures import DistanceRelativeStructure
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

  def fast_angle(self, distances, x, y, z):
    c = distances[:, x, z]
    a = distances[:, x, y]
    b = distances[:, y, z]
    angle = torch.acos((a ** 2 + b ** 2 - c ** 2) / (2 * a * b))
    
    inds = torch.arange(distances.size(-1))
    angle[torch.isnan(angle)] = 0
    return angle

  def fast_dihedral(self, distances, apex, x_1, x_2, x_3):
    phi_12 = self.fast_angle(distances, x_1, apex, x_2)
    phi_13 = self.fast_angle(distances, x_1, apex, x_3)
    phi_23 = self.fast_angle(distances, x_2, apex, x_3)
    phi = torch.acos(((phi_13.cos() - phi_12.cos() * phi_23.cos()) / (phi_12.sin() * phi_23.sin())).clamp(-1, 1))
    
    phi[torch.isnan(phi)] = 0
    
    return phi

  def __getitem__(self, index):
    # Extract the boundaries of a whole protein
    window = slice(self.index[index], self.index[index + 1])
    seq_len = window.stop - window.start

    # Make me a mask
    # Predict at least 5% of the sequence up to the whole seq
    mask = np.random.choice(seq_len, size=np.random.randint(1, seq_len), replace=False)
    mask = torch.tensor(mask, dtype=torch.long)
    mask_binary = torch.zeros(seq_len, dtype=torch.uint8)
    mask_binary[mask] = 1

    # Get sequence info
    primary = self.pris[window] - 1

    primary_masked = primary.clone()
    primary_masked[mask] = 20
    primary_onehot = torch.zeros((seq_len, 21), dtype=torch.float)
    primary_onehot[torch.arange(seq_len), primary_masked] = 1

    # Prepare neighborhood structure
    inds = self.inds[window]
    neighbours = ConstantStructure(0, 0, (inds - self.index[index]).to(torch.long))

    # Prepare orientation infos
    orientation = self.ors[window, :, :].view(seq_len, -1)

    tertiary = self.ters[:, :, window]
    tertiary, angles = self.backrub(tertiary[[0, 1, 3]].permute(2, 0, 1))
    tertiary = tertiary[:, 1] / 100
    tertiary = tertiary + 0.01 * torch.randn_like(tertiary) # corruption FIXME
    angles = angles.transpose(0, 1)

    indices = torch.tensor(
      range(window.start, window.stop),
      dtype=torch.float
    )
    relative_indices = indices[None, :] - indices[:, None]
    relative_sin = (relative_indices / 10).sin()
    relative_cos = (relative_indices / 10).cos()
    
    distances = (tertiary[None, :, :] - tertiary[:, None, :]).norm(dim=-1)
    distances = distances.unsqueeze(0)
    
    inds = torch.arange(distances.size(-1))
    idx_a = inds[:, None]
    idy_a = inds[None, :]
    chain_angle = self.fast_angle(distances, idx_a - 1, idx_a, (idx_a + 1) % distances.size(-1))[:, :, 0].permute(1, 0)
    chain_dihedral = self.fast_dihedral(distances, idx_a - 1, idx_a - 2, idx_a, (idx_a + 1) % distances.size(-1))[:, :, 0].permute(1, 0)
    contact_angles = self.fast_angle(distances, idx_a - 1, idx_a, idy_a)
    contact_dihedrals = self.fast_dihedral(distances, idx_a, idx_a - 1, idy_a, idy_a - 1)
    into_contact_dihedrals = self.fast_dihedral(distances, idx_a - 1, idx_a - 2, idx_a, idy_a)

    angle_features = torch.cat((chain_angle.sin(), chain_angle.cos(), chain_dihedral.sin(), chain_dihedral.cos()), dim=1)

    orientation = torch.cat((distances, contact_angles.sin(), contact_angles.cos(), contact_dihedrals.sin(), contact_dihedrals.cos(), into_contact_dihedrals.sin(), into_contact_dihedrals.cos(), relative_sin[None], relative_cos[None]), dim=0)
    orientation_slice = orientation[:, torch.arange(neighbours.connections.size(0))[:, None], neighbours.connections].permute(1, 2, 0).contiguous()

    # Prepare local features
    dmap = (tertiary[None, :] - tertiary[:, None]).norm(dim=-1)
    closest = torch.arange(tertiary.size(0))
    closest = abs(closest[None, :] - closest[:, None]).topk(15, dim=1).indices
    local_features = dmap[torch.arange(dmap.size(0))[:, None], closest] / 100

    protein = SubgraphStructure(torch.zeros_like(inds))

    features = torch.cat((angle_features, primary_onehot), dim=1)

    inputs = (
      PackedTensor(features),
      PackedTensor(primary),
      PackedTensor(orientation_slice),
      neighbours, protein
    )

    targets = (
      PackedTensor(primary, split=False),
      PackedTensor(mask_binary, split=False)
    )

    return inputs, targets

  def __len__(self):
    return ProteinNet.__len__(self)


class CondSeqResampleNet(CondTransformerNet):
  """
  Similar to CondTransformerNet but with resampling of the sequence based on evolutionary information
  """
  def __init__(self, *args, desired_resample=0.15, **kwargs):
    super(CondSeqResampleNet, self).__init__(*args, **kwargs)
    self.desired_resample = desired_resample

  def __getitem__(self, index):
    # Extract the boundaries of a whole protein
    window = slice(self.index[index], self.index[index + 1])
    seq_len = window.stop - window.start

    # Make me a mask (used for the conditional training)
    # Predict at least 5% of the sequence up to the whole seq
    mask = np.random.choice(seq_len, size=np.random.randint(seq_len // 20, seq_len), replace=False)
    mask = torch.tensor(mask, dtype=torch.long)
    mask_binary = torch.zeros(seq_len, dtype=torch.uint8)
    mask_binary[mask] = 1

    # Get sequence info
    primary = (self.pris[window] - 1).clone()

    # Do the resampling
    to_resample = np.random.choice(seq_len, size=(self.desired_resample * seq_len.float()).long().numpy(), replace=False)
    pssm = self.evos[:20,window][:,to_resample]

    try:
      resampled = torch.multinomial(pssm.t(), 1).flatten()
      primary[to_resample] = resampled
    except:
      pass # TODO

    # Run the masking for conditional transformer training
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
    orientation = torch.cat((distances, indices), dim=1)

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


class MaskedLoss(nn.Module):
  def __init__(self):
    super().__init__()
    self.loss = nn.CrossEntropyLoss()

  def forward(self, inputs, targets):
    targ, mask = targets
    aas = targ[mask]
    unique, count = aas.unique(return_counts=True)
    weights = torch.ones(20, device=inputs.device)
    weights[unique] = 1 / count.float()
    loss = nn.CrossEntropyLoss(weight=weights)
    return self.loss(inputs[mask], targ[mask])
    # except:
    #   return torch.tensor(0, dtype=inputs.dtype, device=inputs.device,)

class SubsampledLoss(MaskedLoss):
  def forward(self, inputs, targets):
    targ, mask = targets
    aas = targ[mask]
    ins = inputs[mask]
    unique, count = aas.unique(return_counts=True)
    resampled_inputs = []
    resampled_targets = []
    for aa in unique:
      aa_mask = aas == aa
      aa_inputs = ins[aa_mask]
      resample = torch.randint(aa_inputs.size(0), (100,))
      aa_targets = aas[aa_mask]
      aa_inputs = aa_inputs[resample]
      aa_targets = aa_targets[resample]
      resampled_inputs.append(aa_inputs)
      resampled_targets.append(aa_targets)
    resampled_inputs = torch.cat(resampled_inputs, dim=0)
    resampled_targets = torch.cat(resampled_targets, dim=0)
    return self.loss(resampled_inputs, resampled_targets)

class ConditionalStructuredTransformerTraining(SupervisedTraining):
  """
  Train a transformer model mapping structure to sequence for each position as a conditional probability of its surroundings
  """
  def each_step(self):
    super().each_step()
    # Schedule from 'Attention is all you need'
    super(ConditionalStructuredTransformerTraining, self).each_step()
    learning_rate = torch.pow(torch.tensor(128.0), -0.5)
    step_num = torch.tensor(float(self.step_id + 1))
    
    self.net.schedule = min(5, self.step_id // 10_000)

    #learning_rate *= min(
    #  torch.pow(step_num, -0.5),
    #  step_num * torch.pow(torch.tensor(4000.0), -1.5)
    #)
    #self.optimizer.param_groups[0]["lr"] = learning_rate

if __name__ == "__main__":
  data = CondTransformerNet(sys.argv[1], num_neighbours=15, n_backrub=20)
  valid_data = CondTransformerNet(sys.argv[2], num_neighbours=15, n_backrub=0) # Validation with out augmentation
  net = SDP(
    ScheduledPrestructuredTransformer(
    4, 128, 9, 
    attention_size=128, heads=8,
    mlp_depth=2, depth=6, decoder_depth=1, schedule=0, batch_norm=True
  ))
  training = ConditionalStructuredTransformerTraining(
    net, data, valid_data,
    [SubsampledLoss()],
    batch_size=32,
    max_epochs=1000,
    optimizer=lambda x: torch.optim.Adam(x), # LR scheduled 
    device="cuda:0",
    network_name="sched-structured-transformer/15-6-1-drop-10-rub-10-prestructured-scheduled-resampled",
    valid_callback=valid_callback,
    report_interval=10
  )
  final_net = training.train()
