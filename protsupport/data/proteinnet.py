import random
import numpy as np
import torch
from torch.utils.data import Dataset

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
