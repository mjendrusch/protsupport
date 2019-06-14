import numpy as np

def compute_dihedral_angle(a, b, c):
  c1 = np.cross(a, b)
  c2 = np.cross(b, c)
  b0 = b / np.linalg.norm(b)
  return np.arctan2(np.cross(c1, c2).dot(b0), c1.dot(c2))

def compute_rotation_matrix(axis, angle):
  axis = axis / np.linalg.norm(axis)
  matrix = np.array([
    [0, -axis[2], axis[1]],
    [0, 0, -axis[0]],
    [0, 0, 0]
  ])
  matrix = matrix - matrix.T
  rot = np.eye(3) + np.sin(angle) * matrix + (1 - np.cos(angle)) * (matrix @ matrix)
  return rot

def compute_rotation(vector, axis, angle):
  rot = compute_rotation_matrix(axis, angle)
  return rot @ vector

def vector_angle(v1, v2):
  v1 = v1 / np.linalg.norm(v1)
  v2 = v2 / np.linalg.norm(v2)
  return np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))
