import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from . import quantizer

from configs.config import *

args = None


class Linear_Q(nn.Linear):
    def __init__(self, *kargs, **kwargs):
        super(Linear_Q, self).__init__(*kargs, **kwargs)


def linear_quantize_fn(bit_list):
    class Linear_Q_(Linear_Q):
        def __init__(self, in_features, out_features, bias=True):
            super(Linear_Q_, self).__init__(in_features, out_features, bias)
            self.bit_list = bit_list
            self.w_bit = self.bit_list[-1]
            self.quantize_fn = quantizer.weight_quantize_fn(self.bit_list)

        def forward(self, input):
            weight_q = self.quantize_fn(self.weight)
            return F.linear(input, weight_q, self.bias)

    return Linear_Q_