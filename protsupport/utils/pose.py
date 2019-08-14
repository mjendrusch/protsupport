import numpy as np
import torch

from pyrosetta import *

from torchsupport.modules.basic import one_hot_encode

from protsupport.data.scripts.process_proteinnet import AA_ID_DICT, compute_dihedrals

def pose_to_net(pose: Pose):
  length = pose.total_residue()
  positions = torch.zeros(4, 3, length)
  sequence = pose.sequence()
  sequence_one_hot = one_hot_encode(
    sequence,
    sorted(list(AA_ID_DICT.keys()), key=AA_ID_DICT.get)
  )
  angles = torch.zeros(3, length)
  mask = torch.ones(length)
  for idx, residue in enumerate(pose.residues):
    nn = residue.atom("N").xyz()
    ca = residue.atom("CA").xyz()
    if residue.aa() == rosetta.core.chemical.AA.aa_gly:
      cb = residue.atom("1HA").xyz()
    else:
      cb = residue.atom("CB").xyz()
    co = residue.atom("C").xyz()
    positions[:, :, idx] = torch.tensor([nn, ca, cb, co])
    
  #   phi = pose.phi(idx + 1) / 180 * np.pi
  #   psi = pose.psi(idx + 1) / 180 * np.pi
  #   omega = pose.omega(idx + 1) / 180 * np.pi
  #   angles[:, idx] = torch.tensor([phi, psi, omega])
  # angles = angles.t().contiguous().view(-1).roll(1).view(-1, 3).t().contiguous()
  angles, _ = compute_dihedrals(
    positions[[0, 1, 3]].numpy().transpose(2, 0, 1).reshape(-1, 3),
    torch.ones(positions.size(-1))
  )
  angles = torch.tensor(angles, dtype=torch.float)
  return positions, angles, sequence_one_hot, mask

def net_to_pose(sequence, positions, angles, mask):
  pose = pose_from_sequence(sequence, "centroid")
  for idx, (phi, psi, omega) in angles.t():
    pose.set_phi(idx + 1, phi / np.pi * 180)
    pose.set_psi(idx + 1, psi / np.pi * 180)
    pose.set_omega(idx + 1, omega / np.pi * 180)
