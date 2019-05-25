import sys
import os

import numpy as np

import ray

from protsupport.data.process_msa import (
  process_fasta
)

@ray.remote
def process_inner(idx, record, msa_args=None, couple_args=None):
  tmp_name = f"tmp.{idx}.fasta"
  with open(tmp_name, "w") as tmp:
    tmp.write("\n".join(record))
  f_i, h_i, f_ij, J_ij = process_fasta(tmp_name, msa_args=msa_args, couple_args=couple_args)
  np.save(f"{tmp_name}.fi", f_i)
  np.save(f"{tmp_name}.hi", h_i)
  np.save(f"{tmp_name}.fij", f_ij)
  np.save(f"{tmp_name}.Jij", J_ij)
  os.remove(tmp_name)
  os.remove(f"{tmp_name}.phi")

def process_all(pdb_path, **kwargs):
  result = []
  with open(pdb_path) as pdb:
    record = [None, None]
    accumulate = ""
    count = -1
    for line in pdb:
      if line.startswith(">"):
        record[0] = line.strip()
        record[1] = accumulate
        accumulate = ""
        if count != -1:
          result.append(process_inner.remote(count, record, **kwargs))
        count += 1
      else:
        accumulate += line.strip()
  return result

pdb_path, hhblits = sys.argv[1:3]

ray.init(num_cpus=2)
ray.get(process_all(
  pdb_path,
  msa_args={
    "db_path" : hhblits
  }
))
