import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


def xor_gate(A,B):
  return (A*(1-B) + (1-A)*B)

def LCAFA1(array):
  print("array:", array)
  A,B,Ci = array.split(1)
  sum = xor_gate(Ci,xor_gate(A,B))
  cout = Ci

  return sum, cout

def EFA(array):
  A,B,Ci = array.split(1)
  sum = xor_gate(Ci, xor_gate(A,B))
  cout = (A*Ci + A*B + B*Ci)%2

  return sum, cout

def SXAFA(array):
  A,B,Ci = array.split(3)
  sum = xor_gate(A,B)
  cout = A*B + Ci

  return sum, cout