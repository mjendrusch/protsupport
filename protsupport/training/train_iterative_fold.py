import sys
import random
import numpy as np

from matplotlib import pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as func
from torchsupport.training.training import SupervisedTraining

from torchsupport.structured import PackedTensor, ConstantStructure
from torchsupport.structured import DataParallel as SDP

from protsupport.data.fold import FoldNet
from protsupport.modules.fold import IterativeAttentionDistancePredictor

class DoubleFoldNet(FoldNet):
  def __getitem__(self, index):
    inputs, outputs = super().__getitem__(index)
    return inputs, outputs, outputs

def valid_callback(training, data, predictions):
  inputs, labels, _ = data
  predictions = predictions[0][0]
  ind = torch.arange(predictions[3].size(1), dtype=torch.float, device=predictions[3].device)
  mean = predictions[3].softmax(dim=1) * ind[None, :, None, None]
  mean = mean.sum(dim=1, keepdim=True) / 42
  mean = mean.repeat_interleave(3, dim=1)
  predictions = predictions[3].argmax(dim=1).to(torch.float)[:, None]
  pred = predictions.repeat_interleave(3, dim=1) / 42
  lab = labels[3].to(torch.float)[:, None].repeat_interleave(3, dim=1)
  lab = lab / 42
  training.writer.add_images("distances", lab, training.step_id)
  training.writer.add_images("predicted", pred, training.step_id)
  training.writer.add_images("mean", mean, training.step_id)

class TotalLoss(nn.Module):
  def __init__(self):
    super().__init__()
    self.ce = nn.CrossEntropyLoss(reduction="none")
    self.kl = nn.KLDivLoss(reduction="none")

  def forward(self, inputs, targets):
    angle, dihedral, pssm, distances, contact_angles, contact_dihedrals, into_dihedrals, mask = targets
    mask2d = mask[:, None, :] * mask[:, :, None]
    msum = mask.float().sum() + 1
    m2sum = mask2d.float().sum() + 1
    loss = (self.ce(inputs[0], angle) * mask.float()).sum() / msum
    loss += (self.ce(inputs[1], dihedral) * mask.float()).sum() / msum
    loss += (self.kl(inputs[2].log_softmax(dim=1), pssm) * mask[:, None].float()).mean(dim=1).sum() / msum
    loss += (self.ce(inputs[3], distances) * mask2d.float()).sum() / m2sum
    loss += (self.ce(inputs[4], contact_angles) * mask2d.float()).sum() / m2sum
    loss += (self.ce(inputs[5], contact_dihedrals) * mask2d.float()).sum() / m2sum
    loss += (self.ce(inputs[6], into_dihedrals) * mask2d.float()).sum() / m2sum
    loss += (self.ce(inputs[7], angle) * mask.float()).sum() / msum
    loss += (self.ce(inputs[8], dihedral) * mask.float()).sum() / msum
    loss += (self.kl(inputs[9].log_softmax(dim=1), pssm) * mask[:, None].float()).mean(dim=1).sum() / msum
    loss += (self.ce(inputs[10], angle) * mask.float()).sum() / msum
    loss += (self.ce(inputs[11], dihedral) * mask.float()).sum() / msum
    loss += (self.kl(inputs[12].log_softmax(dim=1), pssm) * mask[:, None].float()).mean(dim=1).sum() / msum
    return loss

if __name__ == "__main__":
  num_neighbours = 15
  drop = 0.1 if len(sys.argv) < 4 else float(sys.argv[3])
  name = f"iterative-drop-{drop}-1"
  data = DoubleFoldNet(sys.argv[1], num_neighbours=num_neighbours, pass_mask=True)
  valid_data = DoubleFoldNet(sys.argv[2], num_neighbours=num_neighbours, pass_mask=True)
  net = SDP(IterativeAttentionDistancePredictor(
    pair_depth=4, seq_depth=4, size=256, attention_size=64, value_size=16, drop=drop, iterations=20
  ))
  training = SupervisedTraining(
    net, data, valid_data,
    [TotalLoss(), TotalLoss()],
    batch_size=4,
    max_epochs=1000,
    optimizer=lambda x: torch.optim.Adam(x, lr=1e-3),
    device="cuda:0",
    network_name=f"fold/{name}-1",
    valid_callback=valid_callback
  )
  final_net = training.train()
