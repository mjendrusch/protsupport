import sys
import os
from copy import deepcopy

import numpy as np

from jug import TaskGenerator

from protsupport.data.process_msa import (
  process_fasta
)

def save_crushed_coupling(path, array):
  crush = np.float16(array)
  sub = []
  for i in range(crush.shape[0]):
    for j in range(i + 1, crush.shape[0]):
      sub.append(crush[i, j, None])
  sub = np.concatenate(sub, axis=0)
  np.savez_compressed(path, sub)

def save_crushed_profile(path, array):
  crush = np.float16(array)
  np.savez_compressed(path, crush)

@TaskGenerator
def process_inner(idx, head, tail, msa_args=None, couple_args=None):
  tmp_name = f"tmp.{idx}.fasta"
  name = head[1:].split(" ")[0]
  with open(tmp_name, "w") as tmp:
    tmp.write("\n".join([head, tail]))
  f_i, h_i, f_ij, J_ij = process_fasta(tmp_name, msa_args=msa_args, couple_args=couple_args)
  save_crushed_profile(f"{name}.fi", f_i)
  save_crushed_profile(f"{name}.hi", h_i)
  save_crushed_coupling(f"{name}.fij", f_ij)
  save_crushed_coupling(f"{name}.Jij", J_ij)
  os.remove(tmp_name)
  os.remove(f"tmp.{idx}.hhr")
  os.remove(f"{tmp_name}.psi")
  os.remove(f"{tmp_name}.psi.params")
  return idx

def process_all(pdb_path, **kwargs):
  result = []
  with open(pdb_path) as pdb:
    record = [None, None]
    accumulate = ""
    count = -1
    for line in pdb:
      if line.startswith(">"):
        record[1] = accumulate
        accumulate = ""
        if count != -1:
          process_inner(count, record[0], record[1], **kwargs)
        record[0] = line.strip()
        count += 1
      else:
        accumulate += line.strip()
  return result

def process_reduced(pdb_path, name_list, **kwargs):
  result = []
  with open(pdb_path) as pdb:
    record = [None, None]
    accumulate = ""
    count = -1
    for line in pdb:
      if line.startswith(">"):
        record[1] = accumulate
        accumulate = ""
        if count != -1:
          name, chain = record[0][1:].split(" ")[0].split("_")
          if name in name_list: 
            process_inner(count, record[0], record[1], **kwargs)
        record[0] = line.strip()
        count += 1
      else:
        accumulate += line.strip()
  return result

pdb_path, hhblits, address = sys.argv[1:4]

#name_list = []
#with open(address) as names:
#  for line in names:
#    name_list.append(line.strip())

process_all(pdb_path, msa_args={"db_path": hhblits, "blits_args": "-cpu 8"}, couple_args={"plmc_args": "--ncores 8 --fast -le 16.0 -lh 0.01 -m 100"})
