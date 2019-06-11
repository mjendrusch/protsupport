import numpy as np

AA_ID_DICT = {'A': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'K': 9,
              'L': 10, 'M': 11, 'N': 12, 'P': 13, 'Q': 14, 'R': 15, 'S': 16, 'T': 17,
              'V': 18, 'W': 19,'Y': 20}

def encode_primary_string(primary):
  return list([AA_ID_DICT[aa] for aa in primary])

def read_protein_from_file(file_pointer):
  dict_ = {}
  _dssp_dict = {'L': 0, 'H': 1, 'B': 2, 'E': 3, 'G': 4, 'I': 5, 'T': 6, 'S': 7}
  _mask_dict = {'-': 0, '+': 1}

  while True:
    next_line = file_pointer.readline()
    if next_line == '[ID]\n':
      id_ = file_pointer.readline()[:-1]
      dict_.update({'id': id_})
    elif next_line == '[PRIMARY]\n':
      primary = encode_primary_string(file_pointer.readline()[:-1])
      dict_.update({'primary': primary})
    elif next_line == '[EVOLUTIONARY]\n':
      evolutionary = []
      for residue in range(21): evolutionary.append(
        [float(step) for step in file_pointer.readline().split()])
      dict_.update({'evolutionary': evolutionary})
    elif next_line == '[SECONDARY]\n':
      secondary = list([_dssp_dict[dssp] for dssp in file_pointer.readline()[:-1]])
      dict_.update({'secondary': secondary})
    elif next_line == '[TERTIARY]\n':
      tertiary = []
      # 3 dimension
      for axis in range(3): tertiary.append(
        [float(coord) for coord in file_pointer.readline().split()])
      dict_.update({'tertiary': tertiary})
    elif next_line == '[MASK]\n':
      mask = list([_mask_dict[aa] for aa in file_pointer.readline()[:-1]])
      dict_.update({'mask': mask})
    elif next_line == '\n':
      return dict_
    elif next_line == '':
      return None

def _compute_dihedral_angle(a, b, c):
  c1 = np.cross(a, b)
  c2 = np.cross(b, c)
  b0 = b / np.linalg.norm(b)
  return np.arctan2(np.cross(c1, c2).dot(b0), c1.dot(c2))

def compute_inputs(primary, evolutionary, secondary, tertiary, mask):
  primary = np.array(primary)
  evolutionary = np.array(evolutionary)
  secondary = np.array(secondary)
  tertiary = np.array(tertiary).T
  # print(np.linalg.norm(tertiary[0, :] - tertiary[1, :]))
  mask = np.array(mask)
  return primary, evolutionary, secondary, tertiary, mask

def compute_dihedrals(tertiary, mask):
  angles = np.zeros(len(tertiary), dtype=np.float32)
  angle_mask = np.zeros(len(tertiary), dtype=np.bool)
  for idx, pos in enumerate(tertiary):
    valid = idx // 3 < len(mask) - 1 and mask[idx // 3] and mask[idx // 3 + 1]
    if valid:
      vectors = tertiary[idx + 1:idx + 4] - tertiary[idx:idx + 3]
      vnorm = np.linalg.norm(vectors, axis=1)
      some_equal = (vnorm == 0.0).any()
      if not some_equal:
        angle = _compute_dihedral_angle(*vectors)
        angles[idx + 1] = angle
      valid = valid and not some_equal
      angle_mask[idx + 1] = valid
  angles = angles.reshape(3, -1)
  angle_mask = angle_mask.reshape(3, -1)
  return angles, angle_mask

def _compute_rotation(vector, axis, angle):
  axis = axis / np.linalg.norm(axis)
  matrix = np.array([
    [0, -axis[2], axis[1]],
    [0, 0, -axis[0]],
    [0, 0, 0]
  ])
  matrix = matrix - matrix.T
  rot = np.eye(3) + np.sin(angle) * matrix + (1 - np.cos(angle)) * (matrix @ matrix)
  return rot @ vector

def compute_cb(tertiary, mask):
  tertiary = tertiary.reshape(-1, 3, 3)
  extended_tertiary = np.zeros((tertiary.shape[0], 4, 3))
  for idx, (n, ca, co) in enumerate(tertiary):
    if (n == ca).all():
      mask[idx] = False
    if mask[idx]:
      axis = ca - n
      offset = co - ca
      offset_r = _compute_rotation(
        offset, axis, (4 * np.pi) / 3
      )
      cb = ca + offset_r
      extended_tertiary[idx] = np.array([n, ca, cb, co])
  return extended_tertiary

def protein_to_numpy(file_pointer):
  data = read_protein_from_file(file_pointer)
  if data is None:
    return None
  mask = data["mask"]
  primary = data["primary"]
  evolutionary = data["evolutionary"]
  # secondary = data["secondary"]
  tertiary = data["tertiary"]
  primary, evolutionary, secondary, tertiary, mask = \
    compute_inputs(primary, evolutionary, [], tertiary, mask)
  
  extended_tertiary = compute_cb(tertiary, mask)
  angles, angle_mask = compute_dihedrals(tertiary, mask)
  tertiary = np.transpose(extended_tertiary, (1, 2, 0))
  return mask, primary, evolutionary, secondary, tertiary, angles, angle_mask

def proteins_to_numpy(path, file_pointer):
  all_data = []
  
  prot = protein_to_numpy(file_pointer)
  while prot is not None:
    all_data.append(prot)
    prot = protein_to_numpy(file_pointer)
  
  masks = []
  pris = []
  evos = []
  # secs = []
  ters = []
  angs = []
  amas = []
  index = [0]
  for point in all_data:
    mask, pri, evo, sec, ter, ang, ama = point
    index.append(index[-1] + len(pri))
    masks.append(mask)
    pris.append(pri)
    evos.append(evo)
    # secs.append(sec)
    ters.append(ter)
    angs.append(ang)
    amas.append(ama)

  masks = np.concatenate(masks, axis=0)
  pris = np.concatenate(pris, axis=0)
  evos = np.concatenate(evos, axis=1)
  # secs = np.concatenate(secs, axis=0)
  ters = np.concatenate(ters, axis=2)
  angs = np.concatenate(angs, axis=1)
  amas = np.concatenate(amas, axis=1)
  index = np.array(index)
  np.savez_compressed(
    path,
    masks=masks,
    pris=pris,
    evos=evos,
    # secs=secs,
    ters=ters,
    angs=angs,
    amas=amas,
    index=index
  )

if __name__ == "__main__":
  import sys
  path = sys.argv[1]
  target = path + ".npz"
  with open(path) as data:
    proteins_to_numpy(target, data)
