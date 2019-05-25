import torch
from torch import nn

from torchsupport.modules.basic import OneHotEncoder, one_hot_encode

class OneHotAA(nn.Module):
  """One-hot encoding for amino acid sequences."""
  code = "ACDEFGHIKLMNOPQRSTUVWYBZJX"
  def __init__(self, **kwargs):
    super(OneHotAA, self).__init__(
      OneHotAA.code, **kwargs
    )

def one_hot_aa(data, numeric=False):
  """Encodes a sequence of amino acids into one-hot format."""
  return one_hot_encode(data, OneHotAA.code, numeric=numeric)

class OneHotSecondary(nn.Module):
  """One-hot encoding for 8-sort secondary structure."""
  code = "GHIEBST "
  def __init__(self, **kwargs):
    super(OneHotSecondary, self).__init__(
      OneHotSecondary.code, **kwargs
    )

def one_hot_secondary(data, numeric=False):
  """Encodes a secondary structure into one-hot format."""
  return one_hot_encode(data, OneHotSecondary.code, numeric=numeric)
