import sys
import random

from matplotlib import pyplot as plt

from scipy.spatial import cKDTree

import torch
import torch.nn as nn
import torch.nn.functional as func
from torchsupport.training.training import SupervisedTraining
from torchsupport.structured import PackedTensor, SubgraphStructure, scatter
from torchsupport.modules.basic import one_hot_encode

from protsupport.modules.rgn import ConvolutionalGN
from protsupport.data.proteinnet import ProteinNet

class RGNNet(ProteinNet):
  def __init__(self, path):
    ProteinNet.__init__(self, path)

  def __getitem__(self, index):
    result = super().__getitem__(index)
    primary = result["primary"]
    tertiary = result["tertiary"] / 100
    tertiary = tertiary[[0, 1, 3], :, :].permute(2, 0, 1).contiguous().view(-1, 3)
    angles = result["angles"]
    mask = result["mask"]
    mask = torch.repeat_interleave(mask, 3)

    membership = SubgraphStructure(torch.zeros(primary.size(0)))
    primary_onehot = one_hot_encode(primary - 1, range(20)).t()

    inputs = (
      PackedTensor(primary_onehot, split=False),
      membership
    )

    outputs = (
      PackedTensor(tertiary, split=False), PackedTensor(mask.unsqueeze(1), split=False), membership
    )

    return inputs, outputs, outputs, outputs

  def __len__(self):
    return ProteinNet.__len__(self)

class StructuredMaskedCE(nn.Module):
  def __init__(self):
    super().__init__()
    self.loss = nn.MSELoss(reduction="mean")

  def forward(self, inputs, target):
    target, mask, structure = target
    target = target.view(-1, 3)
    dst = lambda x, y: (x - y).norm(dim=1)
    msk = lambda x, y: (x * y).squeeze(dim=1)
    indices = torch.repeat_interleave(structure.indices, 3)
    mask, _, _, _, _ = scatter.pairwise(msk, mask, indices)
    target, _, _, _, _ = scatter.pairwise(dst, target, indices)
    result = self.loss(mask.float() * inputs, mask.float() * target)
    mask_sum = mask.sum()
    return result * float(mask_sum) / float(mask.numel())

if __name__ == "__main__":
  data = RGNNet(sys.argv[1])
  valid_data = RGNNet(sys.argv[2])
  net = ConvolutionalGN(20)
  training = SupervisedTraining(
    net, data, valid_data,
    [StructuredMaskedCE() for idx in range(3)],
    batch_size=4,
    max_epochs=1000,
    optimizer=lambda x: torch.optim.Adam(x, lr=1e-5),
    device="cpu",
    network_name="cgn",
    valid_callback=lambda x,y,z: x
  )
  final_net = training.train()



