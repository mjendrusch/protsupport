import random
import math

import numpy as np
import torch

#from pyrosetta import *
from pyrosetta.rosetta.protocols.simple_moves.sidechain_moves import JumpRotamerSidechainMover
from pyrosetta.rosetta.core.conformation import get_residue_from_name, get_residue_from_name1
from pyrosetta.rosetta.core.chemical import AA
from pyrosetta import MonteCarlo
from pyrosetta.toolbox import mutate_residue

from torchsupport.modules.basic import one_hot_encode

from protsupport.data.scripts.process_proteinnet import AA_ID_DICT
from protsupport.utils.pose import pose_to_net
from protsupport.utils.geometry import nearest_neighbours

class RotamerMover():
  def sample_position(self, pose):
    return random.choice(range(pose.total_residue()))
  def sample_rotamer(self, pose, position):
    raise NotImplementedError("Abstract")
  def sample_residue(self, pose, position):
    raise NotImplementedError("Abstract")
  def fixed_position(self, idx):
    return False

  def apply(self, pose):
    position = self.sample_position(pose)
    while self.fixed_position(position):
      position = self.sample_position(pose)
    residue_name = self.sample_residue(pose, position)
    mutate_residue(pose, position + 1, residue_name)
    self.sample_rotamer(pose, position)

class GuidedRotamerMover(RotamerMover):
  def __init__(self):
    self.rotamer = JumpRotamerSidechainMover()

  def sample_rotamer(self, pose, position):
    residue = pose.residues[position + 1]
    if residue.aa() in {AA.aa_gly, AA.aa_ala}:
      return
    self.rotamer.make_move(residue)

class NetRotamerMover(GuidedRotamerMover):
  def __init__(self, net, pose, k=15, dropout=0.5):
    super(NetRotamerMover, self).__init__()
    self.net = net
    self.tertiary, self.angles, _, self.mask = pose_to_net(pose)
    self.rotations, self.indices = nearest_neighbours(self.tertiary, k=k)
    self.dropout = dropout
    self.lookup = sorted(list(AA_ID_DICT.keys()), key=AA_ID_DICT.get)
    self.k = k

  def fixed_position(self, idx):
    return False

  def sample_residue(self, pose, position, mask=None, argmax=False):
    inds = self.indices[position]
    rot = self.rotations[position]
    sequence = one_hot_encode(pose.sequence(), self.lookup)
    sequence = sequence[:, inds]
    if mask is not None:
      mask = mask[inds].clone()
    else:
      mask = torch.rand(sequence.size(1)) < self.dropout
    mask[0] = 1
    sequence[:, mask] = 0.0
    sequence = torch.cat((mask.unsqueeze(0).float(), sequence), dim=0)

    tertiary = self.tertiary[:, :, inds].clone()
    tertiary = tertiary - tertiary[0:1, :, 0:1]
    tertiary = torch.tensor(rot, dtype=torch.float) @ tertiary
    tertiary = tertiary.view(-1, tertiary.size(-1)) / 10
    angles = self.angles[:, inds]
    features = torch.cat((
      angles.sin(), angles.cos(), tertiary, sequence
    ), dim=0).unsqueeze(0)

    if argmax:
      logits = self.net(features)
      print(logits)
      print(logits.softmax(dim=1))
      prediction = logits.argmax(dim=1)
      print(self.lookup[prediction[0]], mask, sequence.argmax(dim=0))
      sample = prediction.view(-1)[0]
    else:
      prediction = self.net(features)
      dist = torch.distributions.Categorical(logits=prediction)
      sample = dist.sample()[0]#.argmax(dim=1).view(-1)[0]

    return self.lookup[sample]

class NetPackMover(NetRotamerMover):
  def __init__(self, net, pose, fix=None, glycinate=False, max_iter=100,
               n_moves=1, scorefxn=None, kT=0.1, k=15, dropout=0.5):
    super(NetPackMover, self).__init__(net, pose, k=k, dropout=dropout)
    self.n_moves = n_moves
    self.max_iter = max_iter
    self.kT = kT
    self.fix = fix if fix is not None else []
    self.step = 0
    if glycinate:
      self.dropout = 1.0
      mask = torch.tensor([
        1 - int(self.fixed_position(idx))
        for idx in range(len(pose.sequence()))
      ], dtype=torch.bool)
      for idx, residue in enumerate(pose.residues):
        residue_name = "G"#self.sample_residue(pose, idx, mask=mask, argmax=True)
        if not self.fixed_position(idx):
          mutate_residue(pose, idx + 1, residue_name, pack_radius=10.0, pack_scorefxn=scorefxn)
        mask[idx] = 0
      for idy in range(5):
        for idx, residue in enumerate(pose.residues):
          mask[idx] = 1
          residue_name = self.sample_residue(pose, idx, mask=mask, argmax=True)
          if not self.fixed_position(idx):
            mutate_residue(pose, idx + 1, residue_name, pack_radius=10.0, pack_scorefxn=scorefxn)
          mask[idx] = 0
      self.dropout = dropout
    self.scorefxn = scorefxn
    self.monte_carlo = MonteCarlo(pose, scorefxn, kT)
    self.glycinate = glycinate

  def fixed_position(self, idx):
    return idx in self.fix

  def schedule(self, step):
    return self.kT# * math.exp(-step * math.log(2) / 100)

  def apply(self, pose):
    for _ in range(self.max_iter):
      for idx in range(self.n_moves):
        super(NetPackMover, self).apply(pose)
        print(self.scorefxn.score(pose))
      self.monte_carlo.boltzmann(pose)
      self.monte_carlo.set_temperature(self.schedule(self.step))
      self.step += 1

class AnnealedNetPackMover(NetPackMover):
  def __init__(self, net, pose, kT_high=100.0, kT_low=0.3, history_size=4, **kwargs):
    super(AnnealedNetPackMover, self).__init__(net, pose, kT=kT_high, **kwargs)
    self.kT_high = kT_high
    self.kT_low = kT_low
    self.history_size = history_size
    self.energy_history = torch.zeros(history_size)
    self.jump = 0
  
  def update_history(self, current_energy):
    self.energy_history = self.energy_history.roll(1)
    self.energy_history[0] = current_energy

  def schedule(self, step):
    result = self.kT_high
    if self.jump >= self.history_size:
      average = self.energy_history[1:].mean()
      if self.energy_history[0] - average > -1.0:
        result = self.kT_high
        self.jump = 1
      else:
        result = (self.kT_high - self.kT_low) * np.exp(-self.jump) + self.kT_low
        self.jump += 1
    else:
      result = (self.kT_high - self.kT_low) * np.exp(-self.jump) + self.kT_low
      self.jump += 1
    print(
      self.jump,
      self.energy_history[0] - self.energy_history[1:].mean(),
      result,
      self.monte_carlo.lowest_score()
    )
    return result

  def apply(self, pose):
    for _ in range(self.max_iter):
      self.monte_carlo.set_temperature(self.schedule(self.step))
      for idx in range(self.n_moves):
        NetRotamerMover.apply(self, pose)
        self.monte_carlo.boltzmann(pose)
        self.step += 1
      current_energy = self.scorefxn.score(pose)
      self.update_history(current_energy)

class LikelihoodDesign(NetRotamerMover):
  def __init__(self, net, pose, fix=None, max_iter=100,
               n_moves=1, scorefxn=None, kT=0.1, k=15,
               dropout=0.5):
    super(LikelihoodDesign, self).__init__(net, pose, k=k, dropout=dropout)
    self.n_moves = n_moves
    self.max_iter = max_iter
    self.kT = kT
    self.fix = fix if fix is not None else []
    self.step = 0
    self.dropout = 0.0
    initial_sequence = [char for char in pose.sequence()]
    mask = torch.tensor([
      1 - int(self.fixed_position(idx))
      for idx in range(len(pose.sequence()))
    ], dtype=torch.bool)
    for idx, residue in enumerate(pose.residues):
      residue_name = self.sample_residue(initial_sequence, idx, mask=mask, argmax=True)
      if not self.fixed_position(idx):
        initial_sequence[idx] = residue_name
      mask[idx] = 0
    self.initial_sequence = initial_sequence
    self.dropout = dropout
    self.best = self.score(initial_sequence)
    self.best_sequence = initial_sequence

  def single_score(self, seq, position, mask=None):
    inds = self.indices[position]
    rot = self.rotations[position]
    sequence = one_hot_encode(seq, self.lookup)
    sequence = sequence[:, inds]
    if mask is not None:
      mask = mask[inds].clone()
    else:
      mask = torch.rand(sequence.size(1)) < 0#self.dropout
    mask[0] = 1
    sequence[:, mask] = 0.0
    sequence = torch.cat((mask.unsqueeze(0).float(), sequence), dim=0)

    tertiary = self.tertiary[:, :, inds].clone()
    tertiary = tertiary - tertiary[0:1, :, 0:1]
    tertiary = torch.tensor(rot, dtype=torch.float) @ tertiary
    tertiary = tertiary.view(-1, tertiary.size(-1)) / 10
    angles = self.angles[:, inds]
    features = torch.cat((
      angles.sin(), angles.cos(), tertiary, sequence
    ), dim=0).unsqueeze(0)

    logits = self.net(features)
    result = -logits.softmax(dim=1)[0, AA_ID_DICT[seq[position]] - 1]
    return result

  def score(self, seq, mask=None):
    logprob = 0
    with torch.no_grad():
      for position in range(len(seq)):
        logprob += self.single_score(seq, position, mask=mask)
    return logprob

  def sample_residue(self, seq, position, mask=None, argmax=False):
    inds = self.indices[position]
    rot = self.rotations[position]
    sequence = one_hot_encode(seq, self.lookup)
    sequence = sequence[:, inds]
    if mask is not None:
      mask = mask[inds].clone()
    else:
      mask = torch.rand(sequence.size(1)) < self.dropout
    mask[0] = 1
    sequence[:, mask] = 0.0
    sequence = torch.cat((mask.unsqueeze(0).float(), sequence), dim=0)

    tertiary = self.tertiary[:, :, inds].clone()
    tertiary = tertiary - tertiary[0:1, :, 0:1]
    tertiary = torch.tensor(rot, dtype=torch.float) @ tertiary
    tertiary = tertiary.view(-1, tertiary.size(-1)) / 10
    angles = self.angles[:, inds]
    features = torch.cat((
      angles.sin(), angles.cos(), tertiary, sequence
    ), dim=0).unsqueeze(0)

    if argmax:
      print("FS", features.shape)
      logits = self.net(features)
      print(logits)
      print(logits.softmax(dim=1))
      prediction = logits.argmax(dim=1)
      print(self.lookup[prediction[0]], mask, sequence.argmax(dim=0))
      sample = prediction.view(-1)[0]
    else:
      prediction = self.net(features)
      dist = torch.distributions.Categorical(logits=prediction)
      sample = dist.sample()[0]#.argmax(dim=1).view(-1)[0]

    return self.lookup[sample]

  def metropolis(self, current, proposal):
    log_alpha = - (proposal - current) / self.kT
    alpha = np.exp(log_alpha)
    uniform = random.random()
    accept = uniform < alpha
    return accept

  def proposal(self, sequence):
    position = random.randrange(len(sequence))
    while self.fixed_position(position):
      position = random.randrange(len(sequence))
    residue_name = self.sample_residue(sequence, position)
    sequence[position] = residue_name
    return sequence

  def apply(self, sequence):
    sequence = sequence or self.initial_sequence
    current_energy = self.score(sequence)
    for _ in range(self.max_iter):
      steps = random.randint(1, 10)
      for idx in range(steps):
        new_sequence = self.proposal(sequence)
      new_energy = self.score(new_sequence)
      if self.metropolis(current_energy, new_energy):
        sequence = new_sequence
      if new_energy < self.best:
        self.best = new_energy
        self.best_sequence = new_sequence
        print("".join(self.best_sequence), self.best)
        
    return sequence
