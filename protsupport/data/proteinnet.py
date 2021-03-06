import os
import random
import numpy as np
import torch
from scipy.spatial import cKDTree
from torch.utils.data import Dataset

from torchsupport.modules.basic import one_hot_encode

from protsupport.utils.geometry import compute_rotation_matrix, vector_angle, rectify_tertiary

class ProteinNet(Dataset):
  def __init__(self, path):
    super().__init__()
    self.path = path
    archive = np.load(path)
    self.masks = torch.tensor(archive["masks"].astype("int"), dtype=torch.long)
    self.pris = torch.tensor(archive["pris"], dtype=torch.long)
    self.evos = torch.tensor(archive["evos"], dtype=torch.float32)
    self.ters = torch.tensor(archive["ters"], dtype=torch.float32)
    self.angs = torch.tensor(archive["angs"], dtype=torch.float32)
    self.amas = torch.tensor(archive["amas"].astype("int"), dtype=torch.long)
    self.index = torch.tensor(archive["index"], dtype=torch.long)

  def __len__(self):
    return self.index.size(0) - 1

  def __getitem__(self, index):
    window = slice(int(self.index[index]), int(self.index[index + 1]))
    return {
      "mask": self.masks[window],
      "primary": self.pris[window],
      "evolutionary": self.evos[:, window],
      "tertiary": self.ters[:, :, window],
      "angles": self.angs[:, window],
      "anglemask": self.amas[:, window]
    }

class DistogramNet(ProteinNet):
  def __init__(self, path, max_distance=20,
               distance_bins=64, torsion_bins=10):
    super().__init__(path)
    self.max_distance = max_distance * 100
    self.distance_bins = distance_bins
    self.torsion_bins = torsion_bins

  def getdistance(self, data):
    tertiary = data["tertiary"]
    cb = tertiary[2, :, :]
    distance = torch.norm(cb[:, None, :] - cb[:, :, None], 2, dim=0)
    distance = torch.clamp(distance, 0, self.max_distance)
    distogram = (distance / self.max_distance * self.distance_bins).to(torch.long)
    return distogram[None]

  def getrama(self, data):
    rama = data["angles"]
    ramagram = ((rama + np.pi) / (2 * np.pi) * self.torsion_bins).to(torch.long)
    return ramagram

  def tile(self, data):
    px = data[:, :, None].expand(-1, -1, data.size(1))
    py = data[:, None, :].expand(-1, data.size(1), -1)
    return torch.cat((px, py), dim=0)

  def getfeatures(self, data):
    primary = data["primary"]
    evolutionary = data["evolutionary"]
    position = torch.tensor(range(primary.size(0)), dtype=torch.float32).view(1, -1)
    primary_hot = one_hot_encode(primary, list(range(1, 21)))
    primary = self.tile(primary_hot).to(torch.float)
    evolutionary = self.tile(evolutionary)
    position = self.tile(position).to(torch.float)
    position = torch.cat((
      torch.sin(position[:1] - position[1:] / 250 * np.pi),
      torch.cos(position[:1] - position[1:] / 250 * np.pi)
    ), dim=0)
    return torch.cat((position, primary, evolutionary), dim=0)

  def getmasks(self, data):
    mask = data["mask"][None]
    anglemask = data["anglemask"]
    mask = self.tile(mask)
    mask = mask[None, 0] * mask[None, 1]
    return mask, anglemask

  def __getitem__(self, index):
    data = super().__getitem__(index)
    distogram = self.getdistance(data)
    rama = self.getrama(data)
    features = self.getfeatures(data)
    mask, anglemask = self.getmasks(data)
    return {
      "features": features,
      "distogram": distogram,
      "rama": rama,
      "mask": mask,
      "anglemask": anglemask
    }

class DistogramSlice(DistogramNet):
  def __init__(self, path, max_distance=20,
               distance_bins=64, torsion_bins=10,
               size=64):
    super().__init__(path, max_distance=max_distance,
                     distance_bins=distance_bins,
                     torsion_bins=torsion_bins)
    self.size = size

  def __getitem__(self, index):
    data = DistogramNet.__getitem__(self, index)
    length = data["features"].size(-1)
    while length < self.size + 1:
      data = DistogramNet.__getitem__(self, random.randrange(0, self.__len__()))
      length = data["features"].size(-1)
    offset_x = random.randrange(0, length - self.size)
    offset_y = random.randrange(0, length - self.size)
    x_window = slice(offset_x, offset_x + self.size)
    y_window = slice(offset_y, offset_y + self.size)
    return {
      "features": data["features"][:, x_window, y_window],
      "distogram": data["distogram"][:, x_window, y_window],
      "rama": (data["rama"][:, x_window], data["rama"][:, y_window]),
      "mask": data["mask"][:, x_window, y_window],
      "anglemask": (data["anglemask"][:, x_window], data["anglemask"][:, y_window])
    }

class ProteinNetKNN(ProteinNet):
  def __init__(self, path, num_neighbours=20, n_jobs=1, cache=True):
    super(ProteinNetKNN, self).__init__(path)
    self.num_neighbours = num_neighbours
    self.n_jobs = n_jobs
    cache_path = ".".join(path.split(".")[:-1] + [f"{num_neighbours}.cache.npz"]) 
    cache_exists = os.path.isfile(cache_path)
    if cache_exists:
      self.path = path
      archive = np.load(cache_path)
      self.rots = torch.tensor(archive["rots"], dtype=torch.float, requires_grad=False)
      self.inds = torch.tensor(archive["inds"], dtype=torch.long, requires_grad=False)
      self.keeps = torch.tensor(archive["keeps"], dtype=torch.long, requires_grad=False)
      self.index = torch.tensor(archive["index"], dtype=torch.long, requires_grad=False)
      self.pris = self.pris[self.keeps]
      self.evos = self.evos[:, self.keeps]
      self.ters = self.ters[:, :, self.keeps]
      self.angs = self.angs[:, self.keeps]
    else:
      keeps = []
      neighbour_indices = []
      rotations = []
      count = 0
      new_index = [0]
      for idx in range(ProteinNet.__len__(self)):
        data = ProteinNet.__getitem__(self, idx)
        if data["mask"].sum() < num_neighbours:
          continue
        keep, indices, rotation = self._get_knn_data(data)
        keep += self.index[idx]
        keeps.append(keep)
        neighbour_indices.append(indices + count)
        rotations.append(torch.tensor(rotation, dtype=torch.float))
        count += keep.size(0)
        new_index.append(count)
      keeps = torch.cat(keeps, dim=0)
      self.keeps = keeps
      self.rots = torch.cat(rotations, dim=0)
      self.inds = torch.cat(neighbour_indices, dim=0)
      self.index = torch.tensor(new_index, dtype=torch.long)
      self.pris = self.pris[keeps]
      self.evos = self.evos[:, keeps]
      self.ters = self.ters[:, :, keeps]
      self.angs = self.angs[:, keeps]

      if cache:
        np.savez_compressed(
          cache_path,
          rots=self.rots,
          inds=self.inds,
          index=self.index,
          keeps=self.keeps
        )

  def __len__(self):
    return self.rots.size(0)

  def __getitem__(self, index):
    inds = self.inds[index]
    rot = self.rots[index]
    primary = self.pris[inds]
    evolutionary = self.evos[:, inds]
    tertiary = self.ters[:, :, inds].clone()
    tertiary = tertiary - tertiary[0:1, :, 0:1]
    tertiary = rot @ tertiary
    angles = self.angs[:, inds]
    return {
      "indices": inds,
      "primary": primary,
      "evolutionary": evolutionary,
      "tertiary": tertiary,
      "angles": angles
    }

  def _get_knn_data(self, data):
    mask = data["mask"]
    keep = mask.nonzero().view(-1)

    tertiary = data["tertiary"][:, :, keep]
    positions = tertiary[1, :, :]
    pt = positions.transpose(1, 0)
    tree = cKDTree(pt)
    deltas, indices = tree.query(pt, k=self.num_neighbours, n_jobs=self.n_jobs)
    indices = torch.tensor(indices, dtype=torch.long, requires_grad=False)
    deltas = torch.tensor(deltas, dtype=torch.float, requires_grad=False)
    neighbour_tertiary = tertiary[:, :, indices.view(-1)].view(
      tertiary.size(0), tertiary.size(1), *indices.shape
    )
    neighbour_tertiary = neighbour_tertiary.permute(2, 0, 1, 3)
    rotations = rectify_tertiary(neighbour_tertiary)
    return keep, indices, rotations
