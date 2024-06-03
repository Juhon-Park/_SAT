from configs.config import *
import torch
import torch.nn as nn
import torch.nn.functional as F

from models import *
from quan_ops import *
from models.resnet8 import *

__all__ = ['preresnet20q', 'wideresnet40q', 'resnet8q']

class Activate(nn.Module):
    def __init__(self, bit_list, quantize=True):
        super(Activate, self).__init__()
        self.bit_list = bit_list
        self.abit = self.bit_list[-1]
        self.acti = nn.ReLU(inplace=True)
        self.quantize = quantize
        if self.quantize:
            self.quan = activation_quantize_fn(self.bit_list)

    def forward(self, x):
        if self.abit == 32:
            x = self.acti(x)
        else:
            x = torch.clamp(x, 0.0, 1.0)
        if self.quantize:
            x = self.quan(x)
        return x

class QATModule(nn.Module):
    def __init__(self, model: nn.Module, wbit_list, abit_list):
        super().__init__()
        self.model = model
        self.wbit_list = wbit_list
        self.abit_list = abit_list
        self.first_conv = True

        self.quant_model(self.model, self.wbit_list, self.abit_list)
        if layer_cfg is not None:
            self.choose_layer_conv(self.model, self.wbit_list, self.abit_list, layer_cfg)

    def quant_model(self, model, wbit_list, abit_list):
        
        for name, child_module in model.named_children():
            if isinstance(child_module, (nn.Conv2d)):
                if self.first_conv==True:
                    self.first_conv=False
                    continue
                Conv2d = conv2d_quantize_fn(wbit_list, abit_list)
                original = model._modules[name]
                # bias = True if original.bias is not None else False

                model._modules[name] = Conv2d(
                                                in_channels=original.in_channels, 
                                                out_channels=original.out_channels, 
                                                kernel_size=original.kernel_size, 
                                                stride=original.stride, 
                                                padding=original.padding, 
                                                bias=False
                                                )
            elif isinstance(child_module, (nn.BatchNorm2d)):
                NormLayer = batchnorm_fn(wbit_list)
                original = model._modules[name]
                model._modules[name] = NormLayer(original.num_features)
            elif isinstance(child_module, (nn.ReLU)):
                model._modules[name] = Activate(wbit_list)
            else:
                self.quant_model(child_module, wbit_list, abit_list)

    def choose_layer_conv(self, model, wbit_list, abit_list, custom_config):
        def get_module_by_name(model, name):
            """
            Get module from model using its name.
            """
            nested_names = name.split('.')
            current_module = model
            for nested_name in nested_names:
                current_module = current_module._modules[nested_name]
            return current_module

        conv_modules = list(custom_config.items())
        print(conv_modules)
        
        for (layer_name, conv_name) in conv_modules:
            
            conv_module = get_module_by_name(model, layer_name)
            
            # Define conv2d_quantize_fn function to create new convolutional layer
            Conv2d = conv2d_quantize_fn(wbit_list, abit_list, conv_name)
            
            # Replace the original convolutional layer with the quantized one
            nested_names = layer_name.split('.')
            parent_module = get_module_by_name(model, '.'.join(nested_names[:-1]))
            setattr(parent_module, nested_names[-1], Conv2d(
                in_channels=conv_module.in_channels, 
                out_channels=conv_module.out_channels, 
                kernel_size=conv_module.kernel_size, 
                stride=conv_module.stride, 
                padding=conv_module.padding, 
                bias=False
            ))

    def forward(self, input):
        return self.model(input)


def preresnet20q(wbit_list, abit_list, num_classes):

    model = preresnet20(num_classes=num_classes)

    quant_model = QATModule(model=model, wbit_list=wbit_list, abit_list=abit_list)

    return quant_model


def wideresnet40q(wbit_list, abit_list, num_classes):
    
    model = wideresnet40(num_classes = num_classes)
    quant_model = QATModule(model = model, wbit_list = wbit_list, abit_list = abit_list)
    
    return quant_model


def resnet8q(wbit_list, abit_list, num_classes):
    model = resnet8(num_classes = num_classes)
    quant_model = QATModule(model=model, wbit_list = wbit_list, abit_list = abit_list)

    return quant_model 
