import torch
from torch import nn
from torch.nn import functional as func

from torchsupport.modules.basic import MLP

class Baseline(nn.Module):
  def __init__(self, aa_size=21, in_size=5, hidden_size=100,
               p_depth=4, w_depth=3,
               neighbours=10):
    super(Baseline, self).__init__()
    self.in_size = in_size
    self.probability = MLP(in_size, aa_size, hidden_size=hidden_size, depth=p_depth)
    self.weight = MLP(in_size, 1, hidden_size=hidden_size, depth=w_depth)
    self.out = MLP(aa_size * neighbours, aa_size, hidden_size=[300, hidden_size, hidden_size])

  def forward(self, inputs):
    in_view = inputs.transpose(1, 2).reshape(-1, self.in_size)
    p = func.softmax(self.probability(in_view), dim=1)
    w = self.weight(in_view)
    prod = (p * w)
    cat = prod.reshape(inputs.size(0), -1)
    out = self.out(cat)
    return [out]
