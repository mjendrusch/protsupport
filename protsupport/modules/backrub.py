import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as func

from protsupport.utils.geometry import compute_rotation_matrix, compute_dihedrals

class SingleBackrub():
  def __init__(self, phi, psi, tau):
    self.phi = phi
    self.psi = psi
    self.tau = tau

  def __call__(self, inputs):
    fst = torch.randint(inputs.size(0) - 2, (1,))[0]
    snd = fst + 2

    small = inputs[fst + 1:fst + 3, 1] - inputs[fst:fst + 2, 1]
    large = inputs[snd, 1] - inputs[fst, 1]

    phi = (2 * torch.rand((1,))[0] - 1) * self.phi
    psi = (2 * torch.rand((1,))[0] - 1) * self.psi
    tau = (2 * torch.rand((1,))[0] - 1) * self.tau

    phi_mat = compute_rotation_matrix(small[0].numpy(), phi.numpy())
    psi_mat = compute_rotation_matrix(small[1].numpy(), psi.numpy())
    tau_mat = compute_rotation_matrix(large, tau.numpy())

    changed = inputs[fst:fst + 4].numpy()
    offset = changed[0, 1]
    changed = changed - offset

    changed = changed @ tau_mat
    changed[0:2] = changed[0:2] @ phi_mat
    changed[2:4] = (changed[2:4] - changed[2, 1]) @ psi_mat + changed[2, 1]
    changed = changed + offset

    inputs[fst:fst + 4] = torch.tensor(changed)

    return inputs

class Backrub():
  def __init__(self, n_moves=10, phi=np.pi, psi=np.pi, tau=np.pi):
    self.backrub = SingleBackrub(phi=phi, psi=psi, tau=tau)
    self.n_moves = n_moves

  def __call__(self, inputs):
    out = inputs
    for _ in range(self.n_moves):
      out = self.backrub(out)
    angles, msk = compute_dihedrals(out.reshape(-1, 3).numpy(), np.ones(out.size(0), dtype=np.uint8))
    return out, torch.tensor(angles, dtype=torch.float)
