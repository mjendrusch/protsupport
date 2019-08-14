import torch
from torch import nn
from torch.nn import functional as func

from torchsupport.modules.basic import MLP

class Baseline(nn.Module):
  def __init__(self, aa_size=21, in_size=5, hidden_size=100,
               p_depth=3, w_depth=2,
               neighbours=10):
    super(Baseline, self).__init__()
    self.in_size = in_size
    self.probability = MLP(2 * in_size, aa_size, hidden_size=hidden_size, depth=p_depth, batch_norm=False)
    self.weight = MLP(2 * in_size, 1, hidden_size=hidden_size, depth=w_depth, batch_norm=False)
    self.out = MLP(aa_size * neighbours, aa_size, hidden_size=[300, hidden_size, hidden_size], batch_norm=False)

  def forward(self, inputs):
    # concatenate inputs with the central residue
    inputs = torch.cat((
      inputs, inputs[:, :, 0:1].expand_as(inputs)
    ), dim=1)
    in_view = inputs.transpose(2, 1).reshape(-1, 2 * self.in_size)
    assert((in_view[0] == inputs[0, :, 0]).all())
    p = func.softmax(self.probability(in_view), dim=1)
    w = self.weight(in_view)
    prod = (p * w)
    cat = prod.reshape(inputs.size(0), -1)
    out = self.out(cat)
    return out
