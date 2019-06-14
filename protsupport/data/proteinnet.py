import random
import numpy as np
import torch
from scipy.spatial import cKDTree
from torch.utils.data import Dataset

from protsupport.utils.geometry import compute_rotation_matrix, vector_angle

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
    primary = data["primary"][None]
    evolutionary = data["evolutionary"]
    position = torch.tensor(range(primary.size(1)), dtype=torch.float32).view(1, -1)
    primary = self.tile(primary).to(torch.float)
    evolutionary = self.tile(evolutionary)
    position = self.tile(position).to(torch.float)
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
    data = super().__getitem__(index)
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
  def __init__(self, path, num_neighbours=20, n_jobs=1):
    super(ProteinNetKNN, self).__init__(path)
    self.num_neighbours = num_neighbours
    self.n_jobs = n_jobs
    primary = []
    evolutionary = []
    tertiary = []
    deltas = []
    indices = []
    for idx in range(ProteinNet.__len__(self)):
      data = ProteinNet.__getitem__(self, idx)
      if data["mask"].sum() < num_neighbours:
        continue
      p, e, t, d, i = self._get_knn_data(data)
      primary.append(p)
      evolutionary.append(e)
      tertiary.append(t)
      deltas.append(d)
      indices.append(i)
    self.pris = torch.cat(primary, dim=0)
    self.evos = torch.cat(evolutionary, dim=0)
    self.ters = torch.cat(tertiary, dim=0)
    self.dels = torch.cat(deltas, dim=0)
    self.inds = torch.cat(indices, dim=0)

  def __len__(self):
    return self.pris.size(0)

  def __getitem__(self, index):
    return {
      "primary": self.pris[index],
      "evolutionary": self.evos[index],
      "tertiary": self.ters[index],
      "deltas": self.dels[index],
      "indices": self.inds[index]
    }

  def _get_knn_data(self, data):
    mask = data["mask"]
    keep = mask.nonzero().view(-1)

    tertiary = data["tertiary"][:, :, keep]
    primary = data["primary"][keep]
    evolutionary = data["evolutionary"][:, keep]
    positions = tertiary[1, :, :]
    pt = positions.transpose(1, 0)
    tree = cKDTree(pt)
    deltas, indices = tree.query(pt, k=self.num_neighbours, n_jobs=self.n_jobs)
    indices = torch.tensor(indices, dtype=torch.long, requires_grad=False)
    deltas = torch.tensor(deltas, dtype=torch.float, requires_grad=False)
    neighbour_primary = primary[indices.view(-1)].view(1, *indices.shape)
    neighbour_primary = neighbour_primary.permute(1, 0, 2)
    neighbour_evolutionary = evolutionary[:, indices.view(-1)].view(
      evolutionary.size(0), *indices.shape
    )
    neighbour_evolutionary = neighbour_evolutionary.permute(1, 0, 2)
    neighbour_tertiary = tertiary[:, :, indices.view(-1)].view(
      tertiary.size(0), tertiary.size(1), *indices.shape
    )
    neighbour_tertiary = neighbour_tertiary.permute(2, 0, 1, 3)
    neighbour_tertiary = self._rectify_tertiary(neighbour_tertiary)
    return (
      neighbour_primary,
      neighbour_evolutionary,
      neighbour_tertiary,
      deltas,
      indices
    )

  def _rectify_tertiary(self, tertiary):
    ter_np = tertiary.numpy()
    pivot = np.array([1, 0, 0])
    n_pos = ter_np[:, 0:1, :, 0:1]
    ca_pos = ter_np[:, 1:2, :, 0:1]
    ter_np -= n_pos
    ca_pos = ca_pos[:, 0, :, 0]
    for idx, position in enumerate(ca_pos):
      angle = vector_angle(position, pivot)
      axis = np.cross(position, pivot)
      rot = compute_rotation_matrix(position, axis, angle)
      ter_np[idx, :, :] = rot @ ter_np[idx, :, :]
    return tertiary
