import random

import torch
import torch.nn as nn
import torch.nn.functional as func
from torch.utils.checkpoint import checkpoint

from torchsupport.modules.rezero import ReZero
from torchsupport.modules.unet import UNetBackbone
from torchsupport.structured.modules.materialized_transformer import MaterializedTransformerBlock

class ResBlock(nn.Module):
  def __init__(self, size, N=1, drop=None, kernel_size=15, dilation=1):
    super().__init__()
    self.conv = getattr(nn, f"Conv{N}d")
    if drop:
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
        nn.Dropout(drop),
        self.conv(
          size // 2, size, kernel_size,
          dilation=dilation,
          padding=(kernel_size // 2) * dilation
        )
      )
    else:
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
  def __init__(self, in_size, out_size, depth=10, drop=None):
    super().__init__()
    self.preprocess = nn.Conv1d(in_size, out_size, 1)
    self.blocks = nn.Sequential(*[
      ResBlock(
        out_size, N=1,
        kernel_size=15,
        dilation=2 ** (idx % 5),
        drop=drop
      )
      for idx in range(depth)
    ])

  def forward(self, inputs):
    out = self.preprocess(inputs)
    return self.blocks(out)

class Pairwise(nn.Module):
  def __init__(self, in_size, out_size, depth=100, drop=None):
    super().__init__()
    self.preprocess = nn.Conv2d(in_size, out_size, 1)
    self.blocks = nn.Sequential(*[
      ResBlock(
        out_size, N=2,
        kernel_size=5,
        dilation=2 ** (idx % 5),
        drop=drop
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
               drop=None,
               seq_depth=10,
               pair_depth=100, size=64):
    super().__init__()
    self.sequential = Sequential(in_size, size, depth=seq_depth, drop=drop)
    self.pairwise = Pairwise(2 * (in_size + size), size, depth=pair_depth, drop=drop)
    seq_out_sizes = seq_out_sizes or (36, 36, 20)
    pair_out_sizes = pair_out_sizes or (42, 36, 36, 36)
    pair_proj_sizes = pair_proj_sizes or (36, 36, 20)
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

class UNetDistancePredictor(nn.Module):
  def __init__(self):
    pass # TODO

class MaterializedAttentionDistancePredictor(nn.Module):
  def __init__(self, in_size=20, seq_out_sizes=None,
               pair_out_sizes=None,
               pair_proj_sizes=None,
               kernel_size=1, heads=8,
               drop=None, seq_depth=10,
               attention_size=64,
               value_size=None,
               pair_depth=100, size=64):
    super().__init__()
    self.sequential = Sequential(in_size, size, depth=seq_depth, drop=drop)
    self.blocks = nn.ModuleList([
      MaterializedTransformerBlock(
        size, size, size, size,
        attention_size=attention_size, heads=heads,
        value_size=(value_size or size), kernel_size=kernel_size,
        activation=nn.ReLU(), dropout=drop or 0.1
      )
      for idx in range(pair_depth)
    ])
    self.edge_project = nn.Conv2d(2 * (in_size + size) + 100, size, 1, bias=False)
    seq_out_sizes = seq_out_sizes or (36, 36, 20)
    pair_out_sizes = pair_out_sizes or (42, 36, 36, 36)
    pair_proj_sizes = pair_proj_sizes or (36, 36, 20)
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

  def position_embedding(self, size):
    pos = torch.arange(size, dtype=torch.float)
    pos = abs(pos[None, :] - pos[:, None])
    ind = torch.arange(50, dtype=torch.float)[:, None, None]
    features = pos[None].repeat_interleave(50, dim=0)
    features = features / (10000 ** (2 * ind / 100))
    sin = features.sin()
    cos = features.cos()
    features = torch.cat((sin, cos), dim=0)
    return features[None]

  def tile(self, data):
    pos = self.position_embedding(data.size(-1)).to(data.device)
    pos = pos.expand(data.size(0), *pos.shape[1:])
    x = data[:, :, None, :].repeat_interleave(data.size(-1), dim=2)
    y = data[:, :, :, None].repeat_interleave(data.size(-1), dim=3)
    result = torch.cat((x, y, pos), dim=1)
    return result

  def predict(self, nodes, edges):
    node_predictions = [
      pred(nodes)
      for pred in self.seq_predictions
    ]
    edge_predictions = [
      pred(edges)
      for pred in self.pair_predictions
    ]
    projected_x = edges.mean(dim=-1)
    projected_y = edges.mean(dim=-2)
    x_predictions = [
      pred(projected_x)
      for pred in self.pair_projections
    ]
    y_predictions = [
      pred(projected_y)
      for pred in self.pair_projections
    ]
    return (
      *node_predictions,
      *edge_predictions,
      *x_predictions,
      *y_predictions
    )

  def forward(self, inputs, mask):
    sequential = self.sequential(inputs)
    out = torch.cat((sequential, inputs), dim=1)
    out = self.tile(out)

    nodes = sequential
    edges = self.edge_project(out)

    for block in self.blocks:
      nodes, edges = block(nodes, edges, mask)

    predictions = self.predict(nodes, edges)

    return (predictions,)

class CheckpointAttentionDistancePredictor(MaterializedAttentionDistancePredictor):
  def __init__(self, in_size=20, seq_out_sizes=None,
               pair_out_sizes=None,
               pair_proj_sizes=None,
               kernel_size=3, heads=8,
               drop=None, seq_depth=2,
               attention_size=64,
               value_size=None,
               pair_depth=3, size=64,
               split=4):
    super().__init__(
      in_size=in_size, seq_out_sizes=seq_out_sizes,
      pair_out_sizes=pair_out_sizes,
      pair_proj_sizes=pair_proj_sizes,
      kernel_size=kernel_size, heads=heads,
      drop=drop, seq_depth=seq_depth,
      attention_size=attention_size,
      value_size=(value_size or size),
      pair_depth=pair_depth, size=size
    )
    self.split = split

  def iterate(self, split=0):
    def helper(nodes, edges, mask):
      size = (len(self.blocks) + 1) // self.split
      for block in self.blocks[split * size:split * size + size]:
        nodes, edges = block(nodes, edges, mask)
      return nodes, edges
    return helper

  def forward(self, inputs, mask):
    inputs.requires_grad_(True)
    sequential = self.sequential(inputs)
    out = torch.cat((sequential, inputs), dim=1)
    out = self.tile(out)

    nodes = sequential
    edges = self.edge_project(out)

    for idx in range(self.split):
      nodes.requires_grad_(True)
      edges.requires_grad_(True)
      nodes, edges = checkpoint(
        self.iterate(split=idx), nodes, edges, mask
      )

    predictions = self.predict(nodes, edges)

    return (predictions,)

class MixedDistancePredictor(MaterializedAttentionDistancePredictor):
  def __init__(self, in_size=20, seq_out_sizes=None,
               pair_out_sizes=None,
               pair_proj_sizes=None,
               kernel_size=3, heads=8,
               drop=None, seq_depth=2,
               attention_size=64,
               value_size=None,
               res_depth=5,
               pair_depth=5, size=64,
               split=5):
    super().__init__(
      in_size=in_size, seq_out_sizes=seq_out_sizes,
      pair_out_sizes=pair_out_sizes,
      pair_proj_sizes=pair_proj_sizes,
      kernel_size=kernel_size, heads=heads,
      drop=drop, seq_depth=seq_depth,
      attention_size=attention_size,
      value_size=(value_size or size),
      pair_depth=pair_depth, size=size
    )
    self.res_blocks = nn.ModuleList([
      nn.Sequential(*[
        ResBlock(
          size, N=2,
          kernel_size=5,
          dilation=2 ** (idy % 5),
          drop=drop
        )
        for idy in range(res_depth)
      ])
      for idx in range(pair_depth)
    ])
    self.split = split

  def iterate(self, split=0):
    def helper(nodes, edges, mask):
      size = (len(self.blocks) + 1) // self.split
      for block, edge_block in zip(
          self.blocks[split * size:split * size + size],
          self.res_blocks[split * size:split * size + size]
      ):
        nodes, edges = block(nodes, edges, mask)
        edges = edge_block(edges)
      return nodes, edges
    return helper

  def forward(self, inputs, mask):
    inputs.requires_grad_(True)
    sequential = self.sequential(inputs)
    out = torch.cat((sequential, inputs), dim=1)
    out = self.tile(out)

    nodes = sequential
    edges = self.edge_project(out)

    for idx in range(self.split):
      nodes.requires_grad_(True)
      edges.requires_grad_(True)
      nodes, edges = checkpoint(
        self.iterate(split=idx), nodes, edges, mask
      )

    predictions = self.predict(nodes, edges)

    return (predictions,)

class IterativeAttentionDistancePredictor(MaterializedAttentionDistancePredictor):
  def __init__(self, in_size=20, seq_out_sizes=None,
               pair_out_sizes=None,
               pair_proj_sizes=None,
               kernel_size=3, heads=8,
               drop=None, seq_depth=2,
               attention_size=64,
               value_size=None,
               pair_depth=3, size=64,
               iterations=10):
    super().__init__(
      in_size=in_size, seq_out_sizes=seq_out_sizes,
      pair_out_sizes=pair_out_sizes,
      pair_proj_sizes=pair_proj_sizes,
      kernel_size=kernel_size, heads=heads,
      drop=drop, seq_depth=seq_depth,
      attention_size=attention_size,
      value_size=(value_size or size),
      pair_depth=pair_depth, size=size
    )
    self.iterations = iterations

  def iterate(self, nodes, edges, mask):
    for block in self.blocks:
      nodes, edges = block(nodes, edges, mask)
    return nodes, edges

  def forward(self, inputs, mask):
    inputs.requires_grad_(True)
    sequential = self.sequential(inputs)
    out = torch.cat((sequential, inputs), dim=1)
    out = self.tile(out)

    nodes = sequential
    edges = self.edge_project(out)

    rand_step = random.randrange(self.iterations)

    pre_nodes = pre_edges = None
    for idx in range(self.iterations):
      nodes.requires_grad_(True)
      edges.requires_grad_(True)
      nodes, edges = checkpoint(
        self.iterate, nodes, edges, mask
      )
      if idx == rand_step:
        pre_nodes = nodes
        pre_edges = edges

    pre_predictions = self.predict(pre_nodes, pre_edges)
    predictions = self.predict(nodes, edges)

    return (predictions, pre_predictions)
