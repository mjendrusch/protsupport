import torch
import torch.nn as nn
import torch.nn.functional as func
import torchsupport.structured as ts
from torchsupport.modules.basic import MLP
from torchsupport.modules.recurrent import ConvGRUCell1d

class LocalInteraction(nn.Module):
  def __init__(self):
    super(LocalInteraction, self).__init__()
    self.positions = None

  def aggregate(self, inputs):
    aggregated = func.adaptive_max_pool1d(inputs, 1)
    return aggregated

  def update_global(self, inputs, global_features):
    raise NotImplementedError("Abstract.")

  def update_local(self, inputs, global_update):
    raise NotImplementedError("Abstract.")

  def forward(self, positions, features, global_features):
    self.positions = positions.size(1)
    inputs = torch.cat((positions, features), dim=1)
    aggregated = self.aggregate(inputs)
    global_features, global_update = self.update_global(
      aggregated, global_features
    )
    positions, features = self.update_local(inputs, global_update)
    return positions, features, global_features

class ProximalInteraction(LocalInteraction):
  def __init__(self, radius=8.0):
    super(ProximalInteraction, self).__init__()
    self.radius = radius

  def extract_xyz(self, positions):
    raise NotImplementedError("Abstract.")

  def forward(self, positions, features, global_features):
    xyz = self.extract_xyz(positions)
    xyz_graph, batch_structure = ts.to_graph(xyz)
    inputs = torch.cat((positions, features), dim=1)
    inputs_graph = inputs.view(-1, inputs.size(1))

    proximity = ts.DistanceStructure(xyz_graph, batch_structure, 0, radius=self.radius)
    aggregated = self.aggregate(inputs)
    global_features, global_update = self.update_global(
      aggregated, global_features
    )
    positions_graph, features_graph = self.update_local(
      inputs_graph, global_update, proximity
    )

    positions = positions_graph.view(*positions.size())
    features = features_graph.view(*features.size())

    return positions, features, global_features

class ResidualBlock(nn.Module):
  def __init__(self, size, hidden_size,
                dilation=1, activation=func.elu_):
    super(ResidualBlock, self).__init__()
    batch_norm = nn.BatchNorm1d
    conv = nn.Conv1d
    self.bn = nn.ModuleList([
      batch_norm(size),
      batch_norm(hidden_size),
      batch_norm(hidden_size)
    ])
    self.blocks = nn.ModuleList([
      conv(size, hidden_size, 1),
      conv(hidden_size, hidden_size, 3,
           dilation=dilation, padding=dilation),
      conv(hidden_size, size, 1)
    ])
    self.activation = activation

  def forward(self, inputs):
    out = inputs
    for bn, block in zip(self.bn, self.blocks):
      out = block(bn(self.activation(out)))
    return out + inputs

class ResidualLocalInteraction(LocalInteraction):
  def __init__(self, in_size, hidden_size, depth=4, max_dilation=4):
    super(ResidualLocalInteraction, self).__init__()
    self.global_mlp = MLP(2 * in_size, in_size, hidden_size=in_size)
    self.blocks = nn.ModuleList([
      ResidualBlock(in_size, hidden_size, dilation=2 ** (idx % 4))
      for idx in range(depth)
    ])

  def update_global(self, inputs, global_features):
    combined = torch.cat((inputs, global_features), dim=1)
    update = self.global_mlp(combined)
    result = inputs + update
    return result, result

  def update_local(self, inputs, global_update):
    out = inputs + global_update
    for block in self.blocks:
      out = block(out)
    positions = out[:, :self.positions]
    features = out[:, self.positions:]
    return positions, features

class RecurrentGRULocalInteraction(LocalInteraction):
  def __init__(self, in_size, depth=4, max_dilation=4):
    super(RecurrentGRULocalInteraction, self).__init__()
    self.global_z = nn.Linear(2 * in_size, 2 * in_size)
    self.global_r = nn.Linear(2 * in_size, 2 * in_size)
    self.global_h_x = nn.Linear(2 * in_size, in_size)
    self.global_h_s = nn.Linear(2 * in_size, in_size)
    self.blocks = nn.ModuleList([
      ResidualBlock(2 * in_size, in_size, dilation=2 ** (idx % 4))
      for idx in range(depth)
    ])

    self.project_z = nn.Conv1d(2 * in_size, in_size, 1)
    self.project_r = nn.Conv1d(2 * in_size, in_size, 1)
    self.project_h_x = nn.Conv1d(in_size, in_size, 1)
    self.project_h_s = nn.Conv1d(in_size, in_size, 1)

  def update_global(self, inputs, global_features):
    combined = torch.cat((inputs, global_features), dim=1)
    z = torch.sigmoid(self.global_z(combined))
    r = torch.sigmoid(self.global_r(combined))
    result = z * global_features + (1 - z) * torch.tanh(
      self.global_h_x(inputs) + self.global_h_s(global_features * r)
    )
    return result, result

  def update_local(self, inputs, global_features):
    out = inputs + global_features
    for block in self.blocks:
      out = block(out)
    z = torch.sigmoid(self.project_z(out))
    r = torch.sigmoid(self.project_r(out))

    positions = inputs[:, :self.positions]
    state = inputs[:, self.positions:]
    out_pos = out[:, :self.positions]
    out_state = out[:, self.positions:]

    h = z * state + (1 - z) * torch.tanh(
      self.project_h_x(out_pos) + self.project_h_s(out_state * r)
    )
    x = positions + out_pos

    return x, h

class GRULocalInteraction(LocalInteraction):
  def __init__(self, in_size, hidden_size):
    super(GRULocalInteraction, self).__init__()
    self.global_gru = nn.GRUCell(2 * in_size, hidden_size)
    self.local_gru = nn.ConvGRUCell1d(in_size, hidden_size)
    self.project_local = nn.Conv1d(hidden_size, in_size)
    self.project_global = nn.Conv1d(hidden_size, in_size)

  def update_global(self, inputs, global_features):
    out = self.global_gru(inputs, global_features)
    return out, self.project_global(out)

  def update_local(self, inputs, global_update):
    positions = inputs[:, :self.positions]
    state = inputs[:, self.positions:]
    state = self.local_gru(positions, state + global_update)
    positions = self.project_local(state) + positions
    return positions, state
