import sys
import os
import subprocess

import numpy as np

import torch
from torch import nn
from torch.nn import functional as func

def run_msa(path, db_path="", blits_args=""):
  return subprocess.run(f"hhblits {blits_args} -i {path} -opsi {path + '.psi'} -d {db_path}", shell=True)

def run_couple(path, plmc_args=""):
  return subprocess.run(f"plmc -o {path}.params {plmc_args} {path}", shell=True)

def convert_psi_file(path):
  new_repr = []
  with open(path) as psi:
    for line in psi:
      line_split = [
        part for part in line.strip().split(" ") if part
      ]
      new_repr.append(">" + "\n".join(line_split) + "\n")
  with open(path, "w") as psi:
    for line in new_repr:
      psi.write(line)

def parse_coupling_parameters(path):
  precision = "float32"
  with open(path, "rb") as f:
    L, num_symbols, N_valid, N_invalid, num_iter = (
      np.fromfile(f, "int32", 5)
    )

    theta, lambda_h, lambda_J, lambda_group, N_eff = (
      np.fromfile(f, precision, 5)
    )

    alphabet = np.fromfile(
      f, "S1", num_symbols
    ).astype("U1")

    weights = np.fromfile(
      f, precision, N_valid + N_invalid
    )

    target_seq = np.fromfile(f, "S1", L).astype("U1")
    index_list = np.fromfile(f, "int32", L)

    f_i, = np.fromfile(
      f, dtype=(precision, (L, num_symbols)), count=1
    )

    h_i, = np.fromfile(
      f, dtype=(precision, (L, num_symbols)), count=1
    )

    f_ij = np.zeros(
      (L, L, num_symbols, num_symbols)
    )

    J_ij = np.zeros(
      (L, L, num_symbols, num_symbols)
    )

    for i in range(L - 1):
      for j in range(i + 1, L):
        f_ij[i, j], = np.fromfile(
          f, dtype=(precision, (num_symbols, num_symbols)),
          count=1
        )
        f_ij[j, i] = f_ij[i, j].T

    for i in range(L - 1):
      for j in range(i + 1, L):
        J_ij[i, j], = np.fromfile(
          f, dtype=(precision, (num_symbols, num_symbols)),
          count=1
        )
        J_ij[j, i] = J_ij[i, j].T

  return f_i, h_i, f_ij, J_ij

def process_fasta(path, msa_args=None, couple_args=None):
  if msa_args is None:
    msa_args = {}
  if couple_args is None:
    couple_args = {}
  run_msa(path, **msa_args)
  convert_psi_file(path + ".psi")
  run_couple(path + ".psi", **couple_args)

  fi, fij, h, J = parse_coupling_parameters(path + ".psi.params")

  return fi, fij, h, J
