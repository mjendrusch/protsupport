import os
import random
from collections import namedtuple

import numpy as np
from scipy.spatial import cKDTree

import torch
from torch.utils.data import Dataset

from pyrosetta import pose_from_file
from pyrosetta.rosetta.core.scoring.dssp import Dssp

from torchsupport.modules.structured import connected_entities as ce
# from torchsupport.data.structured import SubgraphData

from protsupport.data.embedding import one_hot_aa, one_hot_secondary

def _sequence_from_pose(pose):
  pass # TODO

def _parse_node_file(path):
  pass # TODO

class PDBData(Dataset):
  def __init__(self, path, transform=lambda x: x):
    self.data = np.array([
      os.path.join(path, name)
      for name in os.listdir(path)
      if name.endswith("pdb.gz") or name.endswith("pdb")
    ])
    self.transform = transform

  def __getitem__(self, index):
    result = pose_from_file(self.data[index])
    transformed = self.transform(result)
    secondary = Dssp(transformed)
    return (
      transformed,
      result.sequence(),
      secondary.get_dssp_unreduced_secstruct()
    )

  def __len__(self):
    return len(self.data)

NumericPose = namedtuple("NumericPose", [
  "backbone_trace", "distance", "hbonds", "phi", "psi", "sequence", "structure"
])

class PDBNumeric(PDBData):
  def __init__(self, path, mode="ca", cache=True, transform=lambda x:x):
    super(PDBNumeric, self).__init__(path, transform=transform)
    self.mode = mode
    self.cache = cache
    self.exists = np.zeros_like(self.data, dtype=bool)

    self.good = []
    for idx, data_path in enumerate(self.data):
      if os.path.isfile(data_path + ".trace.npy"):
        self.exists[idx] = 1
        self.good.append(idx)

    self.sum = len(self.good)

  def _cb_distance(self, pose, sequence):
    n_res = pose.total_residue()
    distance = torch.zeros(n_res, n_res)
    for idx in range(n_res):
      atom_name_x = "HA2" if sequence[idx] == "G" else "CB"
      for idy in range(idx + 1, n_res):
        atom_name_y = "HA2" if sequence[idy] == "G" else "CB"
        vx = pose.residue(idx + 1).atom(atom_name_x).xyz()
        vy = pose.residue(idy + 1).atom(atom_name_y).xyz()
        distance[idx, idy] = distance[idy, idx] = (vx - vy).norm()
    return distance

  def _ca_distance(self, pose, sequence):
    n_res = pose.total_residue()
    distance = torch.zeros(n_res, n_res)
    for idx in range(n_res):
      atom_name_x = "CA"
      for idy in range(idx + 1, n_res):
        atom_name_y = "CA"
        vx = pose.residue(idx + 1).atom(atom_name_x).xyz()
        vy = pose.residue(idy + 1).atom(atom_name_y).xyz()
        distance[idx, idy] = distance[idy, idx] = (vx - vy).norm()
    return distance

  def _hbonds(self, pose, sequence):
    result = torch.zeros(pose.total_residue(), pose.total_residue())
    hbonds = pose.get_hbonds()
    for hbond in hbonds.hbonds():
      donor = hbond.don_res() - 1
      acceptor = hbond.acc_res() - 1
      result[donor, acceptor] = 1
      result[acceptor, donor] = 1
    return result

  def _get_impl(self, index):
    pose, sequence, structure = super().__getitem__(index)
    phi = np.array([pose.phi(idx + 1) for idx in range(pose.total_residue())])
    psi = np.array([pose.psi(idx + 1) for idx in range(pose.total_residue())])
    trace = np.array([
      [*residue.atom("CA").xyz()]
      for residue in pose.residues
    ])
    distance = eval(f"self._{self.mode}_distance")(pose, sequence)
    hbonds = self._hbonds(pose, sequence)
    return NumericPose(
      trace, distance, hbonds, phi, psi, sequence, structure
    )

  def _dump_impl(self, data, index):
    trace, distance, hbonds, phi, psi, sequence, structure = data
    np.save(self.data[index] + ".trace.npy", trace)
    np.save(self.data[index] + ".distance.npy", distance)
    np.save(self.data[index] + ".hbonds.npy", hbonds)
    np.save(self.data[index] + ".phi.npy", phi)
    np.save(self.data[index] + ".psi.npy", psi)
    np.save(self.data[index] + ".sequence.npy", sequence)
    np.save(self.data[index] + ".structure.npy", structure)
    self.exists[index] = True

  def _load_impl(self, index):
    path = self.data[index]
    trace = np.load(path + ".trace.npy")
    distance = np.load(path + ".distance.npy")
    hbonds = np.load(path + ".hbonds.npy")
    phi = np.load(path + ".phi.npy")
    psi = np.load(path + ".psi.npy")
    sequence = str(np.load(path + ".sequence.npy"))
    structure = str(np.load(path + ".structure.npy"))
    return NumericPose(
      trace, distance, hbonds, phi, psi, sequence, structure
    )

  def __getitem__(self, index):
    index = self.good[index % self.sum]
    return self._load_impl(index)
    # if self.cache:
    #   if self.exists[index]:
    #     return self._load_impl(index)
    #   else:
    #     if os.path.isfile(self.data[index] + ".trace.npy"):
    #       self.exists[index] = True
    #       return self._load_impl(index)
    #     result = self._get_impl(index)
    #     self._dump_impl(result, index)
    #     return result
    # else:
    #   return self._get_impl(index)

class PDBSimpleDistogram(PDBNumeric):
  def __init__(self, path, size=64, mode='cb', cache=True):
    super(PDBSimpleDistogram, self).__init__(
      path, mode=mode, cache=cache, transform=lambda x:x
    )
    self.size = size

  def __getitem__(self, index):
    _, distance, _, phi, psi, sequence, structure = super().__getitem__(index)
    x_offset = random.randint(0, len(phi) - self.size)
    y_offset = random.randint(0, len(phi) - self.size)
    distance_crop = distance[
      x_offset:x_offset + self.size,
      y_offset:y_offset + self.size
    ]
    phi_crop_x = phi[x_offset:x_offset + self.size]
    phi_crop_y = phi[y_offset:y_offset + self.size]
    psi_crop_x = psi[x_offset:x_offset + self.size]
    psi_crop_y = psi[y_offset:y_offset + self.size]
    structure_crop_x = structure[x_offset:x_offset + self.size]
    structure_crop_y = structure[y_offset:y_offset + self.size]
    sequence_crop_x = sequence[x_offset:x_offset + self.size]
    sequence_crop_y = sequence[y_offset:y_offset + self.size]
    return (
      (sequence_crop_x, sequence_crop_y),
      (structure_crop_x, structure_crop_y),
      (phi_crop_x, phi_crop_y),
      (psi_crop_x, psi_crop_y),
      distance_crop
    )

class PDBFragment(PDBNumeric):
  def __init__(self, path, size=64, **kwargs):
    super(PDBFragment, self).__init__(path, **kwargs)
    self.size = size

  def __getitem__(self, index):
    _, _, _, phi, psi, sequence, structure = \
      super(PDBFragment, self).__getitem__(index)
    while len(phi) <= self.size:
      print("are we looping?")
      _, _, _, phi, psi, sequence, structure = \
        super(PDBFragment, self).__getitem__(index + 1)
    offset = random.randint(0, len(phi) - self.size)
    phi = phi[offset:offset + self.size]
    psi = psi[offset:offset + self.size]
    sequence = sequence[offset:offset + self.size]
    structure = structure[offset:offset + self.size]
    return phi, psi, sequence, structure

class PDBBaseline(Dataset):
  def __init__(self, data, neighbours=20):
    # TODO : rotation normalise every neigbourhood.
    super(Dataset, self).__init__()
    self.data = data
    self.features = []
    self.target = []

    for trace, _, phi, psi, sequence in data:
      tree = cKDTree(trace)
      _, indices = tree.query(trace, k=neighbours)
      features = torch.cat((trace, phi, psi), dim=1)
      neighbour_features = features[indices]
      self.features.append(neighbour_features)
      self.target.append(one_hot_aa(sequence))
    self.features = torch.cat(self.features, dim=0)
    self.target = torch.cat(self.target, dim=0)

  def __getitem__(self, index):
    return (
      self.features[index],
      self.target[index]
    )

  def __len__(self):
    return self.target.size(0)

class PDBKNN(PDBNumeric):
  def __init__(self, path, mode='ca', num_neighbours=50,
               residue_weights=1, transform=lambda x:x):
    super(PDBKNN, self).__init__(path, mode=mode, transform=transform)
    self.residue_weights = residue_weights

  def __getitem__(self, index):
    distance, hbonds, phi, psi, sequence, structure = \
      super(PDBKNN, self).__getitem__(index)
    residue = self._sample_residue(sequence)
    return residue

class ProteinData():
  def __init__(self, folder):
    self.proteins = []
    self.paths = []
    for path, _, files in os.walk(folder):
      for file_path in files:
        if file_path.endswith("pdb1.gz.nodes.csv"):
          complete_path = os.path.join(path, file_path)
          split = complete_path.split(".")[:-2]
          name = split[-3]
          base = ".".join(split)
          self.proteins.append(name)
          self.paths.append(base)

  def __getindex__(self, idx):
    pose = ...
    return {
      "sequence": _sequence_from_pose(pose),
      "structure": pose,
      "distances": np.load(f"{self.paths[idx]}.distance.npy"),
      "hbonds": np.load(f"{self.paths[idx]}.hbond.npy"),
      "nodes": _parse_node_file(f"{self.paths[idx]}.nodes.csv"),
      "backbone_structure": ce.ConnectionStructure.from_csv(
        f"{self.paths[idx]}.backbone.csv"),
      "hbond_structure": ce.ConnectionStructure.from_csv(
        f"{self.paths[idx]}.hbond.csv"),
      "contact_structure": ce.ConnectionStructure.from_csv(
        f"{self.paths[idx]}.contact.csv")
    }

  def len(self):
    return len(self.proteins)

class ProteinSphereData(ProteinData):
  def __init__(self, folder, radius=32):
    super(ProteinSphereData, self).__init__(folder)
    self.radius = radius

  def __getindex__(self, idx):
    data = super(ProteinSphereData, self).__getindex__(idx)
    nodes = data['nodes']
    distances = data['distances']
    rand_node = random.choice(range(len(nodes)))
    chosen = (distances[rand_node] < self.radius).nonzero()
    backbone = data['backbone_structure'].select(chosen)
    hbonds = data['hbond_structure'].select(chosen)
    contacts = data['contact_structure'].select(chosen)
    nodes = nodes[chosen]
    return {
      "nodes": nodes,
      "backbone_structure": backbone,
      "hbond_structure": hbonds,
      "contact_structure": contacts
    }
