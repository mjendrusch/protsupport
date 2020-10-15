import torch
import torch.nn as nn
import torch.nn.functional as func

from torchsupport.modules.rezero import ReZero

class ResBlock(nn.Module):
  def __init__(self, size, N=1, kernel_size=15, dilation=1):
    super().__init__()
    self.conv = getattr(nn, f"Conv{N}d")
    self.block = nn.Sequential(
      self.conv(
        size, size // 2, kernel_size,
        dilation=dilation,
        padding=(kernel_size // 2) * dilation
      ),
      nn.ReLU(),
      self.conv(
        size // 2, size // 2, 1
      ),
      nn.ReLU(),
      self.conv(
        size // 2, size, kernel_size,
        dilation=dilation,
        padding=(kernel_size // 2) * dilation
      )
    )
    self.zero = ReZero(size)

  def forward(self, inputs):
    out = self.block(inputs)
    out = self.zero(inputs, out)
    return out

class Sequential(nn.Module):
  def __init__(self, in_size, out_size, depth=10):
    super().__init__()
    self.preprocess = nn.Conv1d(in_size, out_size, 1)
    self.blocks = nn.Sequential(*[
      ResBlock(
        out_size, N=1,
        kernel_size=15,
        dilation=2 ** (idx % 5)
      )
      for idx in range(depth)
    ])

  def forward(self, inputs):
    out = self.preprocess(inputs)
    return self.blocks(out)

class Pairwise(nn.Module):
  def __init__(self, in_size, out_size, depth=100):
    super().__init__()
    self.preprocess = nn.Conv2d(in_size, out_size, 1)
    self.blocks = nn.Sequential(*[
      ResBlock(
        out_size, N=2,
        kernel_size=5,
        dilation=2 ** (idx % 5)
      )
      for idx in range(depth)
    ])

  def forward(self, inputs):
    out = self.preprocess(inputs)
    return self.blocks(out)

class DistancePredictor(nn.Module):
  def __init__(self, in_size=20, seq_out_sizes=None,
               pair_out_sizes=None,
               pair_proj_sizes=None,
               seq_depth=10,
               pair_depth=100, size=64):
    super().__init__()
    self.sequential = Sequential(in_size, size, depth=seq_depth)
    self.pairwise = Pairwise(2 * (in_size + size), size, depth=pair_depth)
    seq_out_sizes = seq_out_sizes or (36, 36, 21)
    pair_out_sizes = pair_out_sizes or (42, 36, 36, 36)
    pair_proj_sizes = pair_proj_sizes or (36, 36, 21)
    self.seq_predictions = nn.ModuleList([
      nn.Conv1d(size, out_size, 1)
      for out_size in seq_out_sizes
    ])
    self.pair_predictions = nn.ModuleList([
      nn.Conv2d(size, out_size, 1)
      for out_size in pair_out_sizes
    ])
    self.pair_projections = nn.ModuleList([
      nn.Conv1d(size, out_size, 1)
      for out_size in pair_proj_sizes
    ])

  def tile(self, data):
    x = data[:, :, None, :].repeat_interleave(data.size(-1), dim=2)
    y = data[:, :, :, None].repeat_interleave(data.size(-1), dim=3)
    result = torch.cat((x, y), dim=1)
    return result

  def forward(self, inputs):
    print(inputs.shape)
    sequential = self.sequential(inputs)
    sequential_predictions = [
      pred(sequential)
      for pred in self.seq_predictions
    ]
    out = torch.cat((sequential, inputs), dim=1)
    out = self.tile(out)
    pairwise = self.pairwise(out)
    pairwise_predictions = [
      pred(pairwise)
      for pred in self.pair_predictions
    ]
    pairwise_predictions[0] = (
      pairwise_predictions[0] +
      pairwise_predictions[0].transpose(2, 3)
    ) / 2
    projected_x = pairwise.mean(dim=-1)
    projected_y = pairwise.mean(dim=-2)
    x_predictions = [
      pred(projected_x)
      for pred in self.pair_projections
    ]
    y_predictions = [
      pred(projected_y)
      for pred in self.pair_projections
    ]
    return (
      (*sequential_predictions,
       *pairwise_predictions,
       *x_predictions,
       *y_predictions),
    )
