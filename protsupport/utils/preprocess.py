import gzip

from pyrosetta import *

def delete_residues(pose, residues):
  residues = sorted(residues)
  for idx, residue in enumerate(residues):
    pose.delete_residue_slow(residue - idx)

def strip_ligands(pose):
  to_delete = []
  for idx, residue in enumerate(pose.residues, 1):
    if not residue.is_protein():
      to_delete.append(idx)
  delete_residues(pose, to_delete)

def clean_pdb_gz(path, out=None):
  if out is None:
    out = ".".join(path.split(".")[:-2] + ["clean", "pdb", "gz"])
  with gzip.open(path, 'rt') as pdb:
    with gzip.open(out, 'wt') as pdb_out:
      for line in pdb:
        if line.startswith("ATOM") or line.startswith("TER"):
          pdb_out.write(line)
