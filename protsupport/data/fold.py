import numpy as np

import torch

from protsupport.utils.geometry import orientation
from protsupport.modules.backrub import Backrub
from protsupport.data.proteinnet import ProteinNet, ProteinNetKNN

class FoldNet(ProteinNetKNN):
  def __init__(self, path, num_neighbours=20, n_jobs=1, n_backrub=10, N=200,
               phi=0.2 * np.pi, psi=0.2 * np.pi, tau=0.2 * np.pi, cache=True,
               pass_mask=False):
    super(FoldNet, self).__init__(
      path,
      num_neighbours=num_neighbours,
      n_jobs=n_jobs, cache=cache
    )
    self.pass_mask = pass_mask
    self.N = N
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
    phi = torch.acos((
      (phi_13.cos() - phi_12.cos() * phi_23.cos()) /
      (phi_12.sin() * phi_23.sin())).clamp(-1, 1))
    phi[torch.isnan(phi)] = 0

    return phi

  def bin_distance(self, distance):
    out = (distance * 2).to(torch.long)
    out[out > 40] = 41
    return out

  def bin_angle(self, angle):
    out = (angle % (2 * np.pi)) / (2 * np.pi)
    out = out.to(torch.long)
    return out

  def pad(self, data):
    result = data
    if data.dim() == 3:
      result = torch.zeros(data.shape[0], self.N, self.N, dtype=data.dtype)
      result[:, :data.size(-2), :data.size(-1)] = data
    if data.dim() == 2:
      result = torch.zeros(data.shape[0], self.N, dtype=data.dtype)
      result[:, :data.size(-1)] = data
    return result

  def __getitem__(self, index):
    # Extract the boundaries of a whole protein
    window = slice(self.index[index], min(self.index[index + 1], self.N + self.index[index]))
    seq_len = window.stop - window.start

    # Make me a mask
    # Predict at least 5% of the sequence up to the whole seq
    mask = torch.zeros(self.N, dtype=torch.bool)
    mask[:seq_len] = True

    # Get sequence info
    primary = self.pris[window] - 1
    pssm = self.evos[:20, window]
    #pssm = pssm.softmax(dim=0)

    primary_onehot = torch.zeros((20, self.N), dtype=torch.float)
    primary_onehot[primary, torch.arange(seq_len)] = 1

    # Prepare neighborhood structure
    tertiary = self.ters[:, :, window]
    tertiary, _ = self.backrub(tertiary[[0, 1, 3]].permute(2, 0, 1))
    tertiary = tertiary[:, 1] / 100

    distances = (tertiary[None, :, :] - tertiary[:, None, :]).norm(dim=-1)
    distances = distances.unsqueeze(0)

    inds = torch.arange(distances.size(-1))
    idx_a = inds[:, None]
    idy_a = inds[None, :]
    chain_angle = self.fast_angle(
      distances, idx_a - 1, idx_a, (idx_a + 1) % distances.size(-1)
    )[:, :, 0]
    chain_dihedral = self.fast_dihedral(
      distances, idx_a - 1, idx_a - 2, idx_a, (idx_a + 1) % distances.size(-1)
    )[:, :, 0]
    contact_angles = self.fast_angle(
      distances, idx_a - 1, idx_a, idy_a
    )
    contact_dihedrals = self.fast_dihedral(
      distances, idx_a, idx_a - 1, idy_a, idy_a - 1
    )
    into_contact_dihedrals = self.fast_dihedral(
      distances, idx_a - 1, idx_a - 2, idx_a, idy_a
    )

    inputs = (
      primary_onehot
    )

    if self.pass_mask:
      inputs = (
        inputs, mask
      )

    targets = (
      # sequential
      self.pad(self.bin_angle(chain_angle))[0],
      self.pad(self.bin_angle(chain_dihedral))[0],
      self.pad(pssm),
      # pairwise
      self.pad(self.bin_distance(distances))[0],
      self.pad(self.bin_angle(contact_angles))[0],
      self.pad(self.bin_angle(contact_dihedrals))[0],
      self.pad(self.bin_angle(into_contact_dihedrals))[0],
      # mask
      mask
    )

    return inputs, targets

  def __len__(self):
    return ProteinNet.__len__(self)
