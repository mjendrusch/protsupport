import numpy as np

def compute_dihedral_angle(a, b, c):
  c1 = np.cross(a, b)
  c2 = np.cross(b, c)
  b0 = b / np.linalg.norm(b)
  return np.arctan2(np.cross(c1, c2).dot(b0), c1.dot(c2))

def _assign_psea_aux(distances, dihedral, angle,
                     distance_spec, dihedral_spec, angle_spec):
  good = True
  for idx in range(3):
    target, area = distance_spec[idx]
    fits = target - area < distances[idx + 2] < target + area
    good = good and fits
  target, area = dihedral_spec
  left = ((target - area) + 180) % 360 / 180 * np.pi
  right = ((target + area) + 180) % 360 / 180 * np.pi
  good = good and left < dihedral < right
  target, area = dihedral_spec
  left = ((target - area) + 180) % 360 / 180 * np.pi
  right = ((target + area) + 180) % 360 / 180 * np.pi
  good = good and left < dihedral < right
  return good

def assign_psea(positions):
  offsets = positions[:, 1:] - positions[:, :-1]
  distances = np.linalg.norm(positions[:, :1] - positions, axis=0)
  dihedral = compute_dihedral_angle(*offsets[:-1])
  angle = vector_angle(-offsets[0], offsets[1])

  helix_angle = (89, 12)
  sheet_angle = (124, 14)

  helix_dihedral = (50, 20)
  sheet_dihedral = (-170, 45)

  helix_distances = ((5.5, 0.5), (5.3, 0.5), (6.4, 0.6))
  sheet_distances = ((6.7, 0.6), (9.9, 0.9), (12.4, 1.1))

  is_helix = _assign_psea_aux(
    distances, dihedral, angle,
    helix_distances, helix_dihedral, helix_angle
  )
  if is_helix:
    return np.array([1, 0, 0])

  is_sheet = _assign_psea_aux(
    distances, dihedral, angle,
    sheet_distances, sheet_dihedral, sheet_angle
  )
  if is_sheet:
    return np.array([0, 1, 0])
  return np.array([0, 0, 1])

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
  dot = np.dot(v1, v2)
  cross = np.linalg.norm(np.cross(v1, v2))
  return np.arctan2(cross, dot)
