import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from . import quantizer
from quan_ops.acti_quan_ops import *
from quan_ops.bn_quan_ops import *
from quan_ops.approx_adder import *
from configs.config import *
import math


args = None

def str_to_function(functionname):
    return getattr(sys.modules[__name__], functionname)

def conv2d_quantize_fn(wbit_list, abit_list, choose_CONV=None):
    if choose_CONV:
        quan_fn = str_to_function(choose_CONV)
    else:
        quan_fn = str_to_function(args.CONV)
    return quan_fn(wbit_list, abit_list)

class Conv2d_Q(nn.Conv2d):
    def __init__(self, *kargs, **kwargs):
        super(Conv2d_Q, self).__init__(*kargs, **kwargs)

def conv2dQ(wbit_list, abit_list):
    class Conv2d_Q_(Conv2d_Q):
        def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1,
                     bias=True):
            super(Conv2d_Q_, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups,
                                            bias)
            self.wbit_list = wbit_list
            self.wbit = self.wbit_list[-1]
            self.quantize_fn = quantizer.weight_quantize_fn(self.wbit_list)

        def forward(self, input, order=None):
            weight_q = self.quantize_fn(self.weight)
            return F.conv2d(input, weight_q, self.bias, self.stride, self.padding, self.dilation, self.groups)

    return Conv2d_Q_


class Conv2d_Q_LR(nn.Module):
    def __init__(self, *kargs, **kwargs):
        super(Conv2d_Q_LR, self).__init__(*kargs, **kwargs)

def conv2dQ_lr(wbit_list, abit_list):
    class Conv2d_Q_LR_(Conv2d_Q_LR):
        def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=False):
            super(Conv2d_Q_LR_, self).__init__()
            self.stride = stride
            self.rank = in_channels // args.rank
            self.groups = args.groups
            # print(LR_cfg.rank, LR_cfg.groups)
            norm_layer = batchnorm_fn(wbit_list)
            self.act = activation_quantize_fn(abit_list)

            conv2d = conv2dQ(wbit_list, abit_list)  
            self.conv_R = conv2d(in_channels=in_channels, 
                                out_channels=self.rank*self.groups, 
                                kernel_size=kernel_size, 
                                stride=stride, 
                                padding=padding, 
                                bias=False, 
                                groups=self.groups)
            self.bn = norm_layer(self.rank*self.groups)
            self.conv_L = conv2d(in_channels=self.rank*self.groups, 
                                out_channels=out_channels, 
                                kernel_size=1, 
                                stride=1, 
                                padding=0, 
                                bias=False)
            

        def forward(self, x):
            x = self.conv_R(x)
            x = self.bn(x)
            x = self.act(x)
            x = self.conv_L(x)
            return x
        
    return Conv2d_Q_LR_




class Conv2d_NoQ(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super(Conv2d_NoQ, self).__init__(*args, **kwargs)

def conv2d_noQ(wbit_list, abit_list):
    class Conv2d_NoQ_(Conv2d_NoQ):
        def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1,
                     bias=True):
            super(Conv2d_NoQ_, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups,
                                            bias)
          

        def forward(self, input):
            return F.conv2d(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

    return Conv2d_NoQ_


class Conv2d_Sparse(nn.Conv2d, nn.Module):
    def __init__(self, *kargs, **kwargs):
        super(Conv2d_Sparse, self).__init__(*kargs, **kwargs)

def conv2d_sparse(wbit_list, abit_list):
    class Conv2d_Sparse_(Conv2d_Sparse):
        def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1,
                     bias=True, *args, **kwargs):
            super(Conv2d_Sparse_, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

            self.kernel_size = (kernel_size,kernel_size)
            self.out_channels = out_channels
            self.padding = (padding, padding)
            
            
            # Initialize weights 
            self.weight = nn.Parameter(torch.randint(0,2,(out_channels, in_channels, self.kernel_size[0],
                                                                self.kernel_size[0])).float().cuda())

            self.pw_width = 3
            self.pw_height = 3

            #flattened weight
            if self.kernel_size[0] == 3:
                self.weight_map = torch.nn.Parameter(self._gen_SDK_mapping(self.weight)).cuda()
            else:
                self.weight_map = torch.squeeze(self.weight).cuda()
            


        def _ordered_pairs_sum(self, x):
            a = torch.arange(x + 1)
            b = x - a
            pairs = torch.stack((a, b), dim=1)
            return pairs

        def _gen_SDK_mapping(self, my_tensor):
            h_diff = self.pw_height - self.kernel_size[0]
            w_diff = self.pw_width - self.kernel_size[0]

            ver_pads = self._ordered_pairs_sum(h_diff)
            hor_pads = self._ordered_pairs_sum(w_diff)

            SDK_mapping = []

            for i in range(len(ver_pads)):
                for j in range(len(hor_pads)):
                    p2d = (hor_pads[j,0], hor_pads[j,1], ver_pads[i,0], ver_pads[i,1])
                    padded_kernel =  F.pad(my_tensor, p2d, mode='constant', value=0)
                    flat_kernel = padded_kernel.view(self.out_channels, -1)

                    SDK_mapping.append(flat_kernel)

            SDK_mapping = torch.concat(SDK_mapping)

            return SDK_mapping

        #partial product
        def _forward(self, x):

            weight = self.weight_map.detach().unsqueeze(1)
            x = x.unsqueeze(1)
            weight = weight
            x = x
            partial_product = torch.mul(x,weight)
            return partial_product

        #flattened input & AT1, AT2 & sum & reshape
        def _slice_and_forward(self, x):
            num, depth, height, width = x.shape

            stride_ver = self.pw_height - self.kernel_size[0] + 1
            stride_hor = self.pw_width  - self.kernel_size[0] + 1

            pad_ver = (height + 2 - self.pw_height) % stride_ver
            pad_hor = (width  + 2 - self.pw_width)  % stride_hor
            

            slide_ver = math.ceil((height + 2 - self.pw_height) / stride_ver) + 1
            slide_hor = math.ceil((width  + 2 - self.pw_width ) / stride_hor) + 1

            if self.padding[0]:
                padded_x = F.pad(x, (1, 1 + pad_hor, 1, 1 + pad_ver),mode='constant', value=0)
                
            else:
                padded_x = x
                stride_ver = 1
                stride_hor = 1

            flattened_input = F.unfold(padded_x,
                                    kernel_size= self.kernel_size,
                                    stride=(stride_ver, stride_hor)).transpose(1,2).squeeze()
            flattened_input = torch.where(flattened_input > 0.0, torch.tensor(1.0, device=flattened_input.device), torch.tensor(0.0, device=flattened_input.device))


            lin_out = self._forward(flattened_input).detach()
            
            weight_size = lin_out.shape[3]
            
            lin_out = lin_out.reshape((-1,weight_size))
            lin_out = torch.transpose(lin_out,0,1)
            lin_out = self._aft_sat(lin_out)                       
            lin_out = self._aft_sec_sat(lin_out)


            if self.padding[0]:
                lin_out = lin_out.squeeze(1)
            
            lin_out = lin_out[0][0] + lin_out[0][1]*2 +lin_out[1][0]*2 + lin_out[1][1]*4

            lin_out = torch.sum(lin_out, dim=0).detach()
            lin_out = lin_out.view(1,-1)
            
            if not self.padding[0]:
                slide_ver = height - self.kernel_size[0] +1
                slide_hor = width  - self.kernel_size[0] +1


            lin_out = lin_out.reshape(num, slide_ver, slide_hor,
                                     1, 1, self.out_channels)

            
            lin_out = lin_out.transpose(2,3)
            lin_out = lin_out.reshape(num,
                                     slide_ver+int(pad_ver/2),
                                     slide_hor+int(pad_hor/2),
                                     self.out_channels)
            lin_out = lin_out.transpose(3,1).transpose(3,2)

            lin_out = lin_out[:,:,:slide_ver,:slide_hor]
            return lin_out
        
        #AT1
        def _aft_sat(self, x):
            device = x.device
            size = x.shape[0]

            #need to deal with the case when if size%3 == (1 or 2). currently just delete the last row
            if size%3:
                x = x[:-(size%3), :]
                
            x = x.reshape(3,-1)
            s, co = EFA(x.to(device))
            s = s.reshape(int(size/3), -1)
            co = co.reshape(int(size/3), -1)

            result = torch.stack((s,co))

            # sum_list = []
            # cout_list = []
            # a = x.split(3)
            # for i in range(int(size / 3)):
            #     s, co = EFA(a[i].detach())
            #     sum_list.append(s)
            #     cout_list.append(co)

            # if size % 3:
            #     sum_list.append(x[int(size / 3) * 3:].detach())
            #     dummy = torch.zeros(x[int(size / 3) * 3:].shape[0], co.shape[1], device=device)
            #     cout_list.append(dummy)

            # sum1 = torch.cat(sum_list, 0)
            # cout = torch.cat(cout_list, 0)

            # result = torch.stack((sum1.detach(), cout.detach()))
            return result.to(device)

        #AT2
        def _aft_sec_sat(self, x):
            sum2 = self._aft_sat(x[0])
            cout2 = self._aft_sat(x[1])
            answer = torch.stack((sum2, cout2))
            return answer.to(x.device)

        def forward(self, input):
            return self._slice_and_forward(input.float())

    return Conv2d_Sparse_
        