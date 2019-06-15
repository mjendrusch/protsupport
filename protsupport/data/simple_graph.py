import os
import numpy as np
from scipy.spatial import cKDTree

from pyrosetta import *

from protsupport.utls.preprocess import strip_ligands, clean_pdb_gz

def get_distance_matrix(pose):
  result = np.zeros((pose.total_residue(), pose.total_residue()))
  for idx in range(1, pose.total_residue()):
    pos1 = get_ca_position(pose.residue(idx))
    for idy in range(idx + 1, pose.total_residue()):
      pos2 = get_ca_position(pose.residue(idy))
      result[idx - 1, idy - 1] = result[idy - 1, idx - 1] = np.linalg.norm(pos1 - pos2)
  return result

def get_hbond_matrix(pose):
  result = np.zeros((pose.total_residue(), pose.total_residue()))
  hbonds = pose.get_hbonds()
  for hbond in hbonds.hbonds():
    donor = hbond.don_res() - 1
    acceptor = hbond.acc_res() - 1
    result[donor, acceptor] = 1
    result[acceptor, donor] = 1
  return result

def get_angle_matrix(pose):
  result = np.zeros((3, pose.total_residue()))
  for idx in range(1, pose.total_residue()):
    result[:, idx] = np.array([
      pose.phi(idx),
      pose.psi(idx),
      pose.omega(idx)
    ])
  return result

def get_backbone_map(pose):
  """Creates an adjacency map for the backbone of a pose."""
  result = [[1]] + [
    [idx - 1, idx + 1]
    for idx in range(1, pose.total_residue() - 1)
  ] + [[pose.total_residue() - 1]]
  return result

def get_ca_position(residue):
  """Computes the C-alpha position of a residue."""
  tmp = residue.atom(2).xyz()
  result = np.array([tmp.x, tmp.y, tmp.z])
  return result

def get_ca_distance_contacts(pose, distance=8):
  """Computes an adjacency map from C-alpha atoms within a given distance."""
  positions = np.array([
    get_ca_position(residue)
    for residue in pose.residues
  ])
  tree = cKDTree(positions)
  return tree.query_ball_tree(tree, distance)

def get_knn_contacts(pose, k=10):
  """TODO"""
  pass

def get_rotamer_contacts(pose, k=10):
  """TODO"""
  pass

def get_contact_map(pose, method='distance', **kwargs):
  """Computes a map of residue contacts."""
  return get_ca_distance_contacts(pose, **kwargs)

def get_hbond_map(pose):
  """Computes a map of residue polar interactions."""
  hbonds = pose.get_hbonds()

  result = [[] for idx in range(pose.total_residue())]
  for hbond in hbonds.hbonds():
    donor = hbond.don_res() - 1
    acceptor = hbond.acc_res() - 1
    result[donor].append(acceptor)
    result[acceptor].append(donor)
  return result

def get_node_properties(pose):
  """Retrieves the properties of all residues in a pose."""
  result = []
  for idx, residue in enumerate(pose.residues, 1):
    data = {
      "kind": residue.name(),
      "pos": get_ca_position(residue),
      "phi": pose.phi(idx),
      "psi": pose.psi(idx),
      "omega": pose.omega(idx)
    }
    result.append(data)
  return result

def get_graph_properties(pose, methods):
  """Computes all requested properties of a pose's graph."""
  strip_ligands(pose)
  nodes = get_node_properties(pose)
  adjacencies = {}
  for method in methods:
    adjacencies[method[0]] = method[1](pose)
  return nodes, adjacencies

def process_graph_properties(path, methods):
  pose = pose_from_file(path)
  nodes, adjacencies = get_graph_properties(pose, methods)
  with open(f"{path}.nodes.csv", "w") as csv_file:
    csv_file.write("kind,x,y,z,phi,psi,omega\n")
    for node in nodes:
      line = ",".join(map(str, [node['kind'], *node['pos'], node['phi'], node['psi'], node['omega']]))
      csv_file.write(
        line + "\n"
      )
  for name in adjacencies:
    values = adjacencies[name]
    with open(f"{path}.{name}.csv", "w") as csv_file:
      for value in values:
        csv_file.write(",".join(map(str, value)) + "\n")
  return pose

def make_data_files(folder):
  for path, name, files in os.walk(folder):
    for file_path in files:
      if file_path.endswith("pdb1.gz"):
        try:
          pose = process_graph_properties(
            os.path.join(path, file_path),
            (
              ("contact", get_contact_map),
              ("hbond", get_hbond_map),
              ("backbone", get_backbone_map)
            )
          )
          np.save(f"{path}/{file_path}.distance.npy", get_distance_matrix(pose))
          np.save(f"{path}/{file_path}.hbond.npy", get_hbond_matrix(pose))
          np.save(f"{path}/{file_path}.angle.npy", get_angle_matrix(pose))
        except:
          pass

def clean_data_files(folder):
  for path, name, files in os.walk(folder):
    for file_path in files:
      clean_pdb_gz(os.path.join(path, file_path))
