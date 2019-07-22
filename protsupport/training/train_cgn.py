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
from torchsupport.structured import DataParallel as SDP

from protsupport.modules.rgn import ConvolutionalGN
from protsupport.data.proteinnet import ProteinNet

class RGNNet(ProteinNet):
  def __init__(self, path):
    ProteinNet.__init__(self, path)

  def __getitem__(self, index):
    result = super().__getitem__(index)
    primary = result["primary"][:2000]
    evolutionary = result["evolutionary"][:2000].t()
    tertiary = result["tertiary"] / 100
    tertiary = tertiary[[0, 1, 3], :, :].permute(2, 0, 1).contiguous()[:2000].view(-1, 3)
    angles = result["angles"][:2000].contiguous().view(-1)
    mask = result["mask"][:2000].view(-1)

    mask = torch.repeat_interleave(mask, 3)

    membership = SubgraphStructure(torch.zeros(primary.size(0), dtype=torch.long))
    primary_onehot = one_hot_encode(primary - 1, range(20)).t()

    inputs = (
      PackedTensor(torch.cat((primary_onehot, evolutionary), dim=1), split=False),
      membership
    )

    outputs = (
      PackedTensor(tertiary, split=False), PackedTensor(mask.unsqueeze(1), split=False), membership
    )

    print("tsize", angles.size())

    return inputs, outputs, PackedTensor(angles, split=False)

  def __len__(self):
    return ProteinNet.__len__(self)

class StructuredMaskedCE(nn.Module):
  def __init__(self):
    super().__init__()
    #self.loss = nn.MSELoss(reduction="sum")

  def forward(self, inputs, target):
    target, mask, structure = target
    target = target.view(-1, 3)
    inputs = inputs.view(-1, 3)
    dst = lambda x, y: (x - y).norm(dim=1)
    msk = lambda x, y: (x * y).squeeze(dim=1)
    indices = torch.repeat_interleave(structure.indices, 3)
    indices = indices[(mask > 0).nonzero().view(-1)]
    target = target[(mask > 0).nonzero().view(-1)]
    inputs = inputs[(mask > 0).nonzero().view(-1)]
    # unique, counts = indices.unique(return_counts=True)
    print("idt", indices.dtype)
    # mask, _ = scatter.pairwise_no_pad(msk, mask, indices)
    target, rmsd_indices = scatter.pairwise_no_pad(dst, target, indices)
    inputs, _ = scatter.pairwise_no_pad(dst, inputs, indices)
    print(target.shape, mask.shape, inputs.shape)
    print("rdt", rmsd_indices.dtype)
    result = (inputs - target) ** 2
    result = torch.sqrt(scatter.mean(result, rmsd_indices) + 1e-6)
    return result.mean()

if __name__ == "__main__":
  data = RGNNet(sys.argv[1])
  valid_data = RGNNet(sys.argv[2])
  net = ConvolutionalGN(41, depth=3)
  training = SupervisedTraining(
    net, data, valid_data,
    [StructuredMaskedCE(), nn.MSELoss()],
    batch_size=1,
    max_epochs=1000,
    optimizer=lambda x: torch.optim.Adam(x, lr=1e-5),
    device="cpu",
    network_name="cgn",
    valid_callback=lambda x,y,z: x
  )
  final_net = training.train()

