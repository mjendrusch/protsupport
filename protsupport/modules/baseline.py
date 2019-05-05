import torch
from torch import nn
from torch.nn import functional as func

from torchsupport.modules.basic import MLP

class Baseline(nn.Module):
  def __init__(self, in_size=5, hidden_size=100,
               p_depth=4, w_depth=3,
               neighbours=10):
    super(Baseline, self).__init__()
    self.probability = MLP(5, 20, hidden_size=100, depth=p_depth)
    self.weight = MLP(5, 1, hidden_size=100, depth=w_depth)
    self.out = MLP(20 * neighbours, 20, hidden_size=[300, 100, 100])

  def forward(self, inputs):
    p = func.softmax(self.probability(inputs), dim=1)
    w = self.weight(inputs)
    cat = (p * w).view(inputs.reshape(inputs.size(0), -1))
    out = self.out(cat)
    return out
