from DeepNodes.layers import *
from DeepNodes.types import *
from DeepNodes.activations import *
import torch
import math
from torch.nn import Parameter, init
import torch.nn.functional as F
from collections import namedtuple

class _ConvNd(Layer):  
    def __init__(self, n, name, input_shape, output_shape, kernel_size, weight, bias, stride, padding, dilation, groups, activation):
        super(Layer, self).__init__()
        if weight == Auto:
            in_channels = input_shape.input[1]
            out_channels = output_shape.output[1]
            if type(kernel_size) == int:
                kernel_size = (kernel_size,)*n
            self.kernel_size = kernel_size
            self.weight = Parameter(torch.Tensor(out_channels, in_channels // groups, *kernel_size))
            init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        else:
            self.weight = Parameter(weight)
            shape = weight.shape
            in_channels = shape[0]
            out_channels = shape[1]*groups
            self.kernel_size = shape[2:]
        self.input_shape = Conv2d.inputs((None,in_channels,*((None,)*n)))
        self.output_shape = Conv2d.outputs((None,out_channels,*((None,)*n)))
        if bias is None:
            self.register_parameter('bias', None)
        elif bias == "auto":
            self.bias = Parameter(torch.Tensor(out_channels))
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)
        else:
            self.bias = bias
        print("stride: "+str(stride))
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        if type(activation)==str:
            self.activation = activations[activation]
        else:
            self.activation = activation


class Conv1d(_ConvNd):
    inputs  = namedtuple('inputs', ['input'])
    outputs = namedtuple('outputs',['output'])
    input_dim  = inputs(3)
    output_dim = outputs(3)
    args = {
            "kernel_size":  (3,     integer),
            "weight":       (Auto,  auto|tensor),
            "bias":         (None,  none|auto|tensor),
            "stride":       (1,     integer),
            "padding":      (0,     integer),
            "dilation":     (1,     integer),
            "groups":       (1,     integer),
            "activation":   ("id",  activation),
    }
    def __init__(self, name, input_shape, output_shape, kernel_size, weight, bias, stride, padding, dilation, groups, activation):
        super().__init__(1, name, input_shape, output_shape, kernel_size, weight, bias, stride, padding, dilation, groups, activation)    
    def forward(self, input):
        return Conv1d.outputs(
            self.activation(
                F.conv1d(input,self.weight,self.bias,self.stride,self.padding,self.dilation,self.groups)))

class Conv2d(_ConvNd):
    inputs  = namedtuple('inputs', ['input'])
    outputs = namedtuple('outputs',['output'])
    input_dim  = inputs(4)
    output_dim = outputs(4)
    args = {
            "kernel_size":  (3,     integer),
            "weight":       (Auto,  auto|tensor),
            "bias":         (None,  none|auto|tensor),
            "stride":       (1,     integer|integer&integer),
            "padding":      (0,     integer),
            "dilation":     (1,     integer),
            "groups":       (1,     integer),
            "activation":   ("id",  activation),
    }
    def __init__(self, name, input_shape, output_shape, kernel_size, weight, bias, stride, padding, dilation, groups, activation):
        super().__init__(2, name, input_shape, output_shape, kernel_size, weight, bias, stride, padding, dilation, groups, activation)    
    def forward(self, input):
        return Conv2d.outputs(
            self.activation(
                F.conv2d(input,self.weight,self.bias,self.stride,self.padding,self.dilation,self.groups)))
        
class Conv3d(_ConvNd):
    inputs  = namedtuple('inputs', ['input'])
    outputs = namedtuple('outputs',['output'])
    input_dim  = inputs(5)
    output_dim = outputs(5)
    args = {
            "kernel_size":  (3,     integer),
            "weight":       (Auto,  auto|tensor),
            "bias":         (None,  none|auto|tensor),
            "stride":       (1,     integer|integer&integer&integer),
            "padding":      (0,     integer),
            "dilation":     (1,     integer),
            "groups":       (1,     integer),
            "activation":   ("id",  activation),
    }
    def __init__(self, name, input_shape, output_shape, kernel_size, weight, bias, stride, padding, dilation, groups, activation):
        super().__init__(3, name, input_shape, output_shape, kernel_size, weight, bias, stride, padding, dilation, groups, activation)    
    def forward(self, input):
        return Conv3d.outputs(
            self.activation(
                F.conv3d(input,self.weight,self.bias,self.stride,self.padding,self.dilation,self.groups)))
                              
convs = {1:Conv1d,2:Conv2d,3:Conv3d}
def ConvNd(name, input_shape, output_shape, kernel_size, weight, bias, stride, padding, dilation, groups, activation):
    return convs[len(input_shape.input)-2](name, input_shape, output_shape, kernel_size, weight, bias, stride, padding, dilation, groups, activation)
ConvNd.inputs  = namedtuple('inputs', ['input'])
ConvNd.outputs = namedtuple('outputs',['output'])
ConvNd.input_dim  = ConvNd.inputs((3,5))
ConvNd.output_dim = ConvNd.outputs((3,5))
ConvNd.args = {
        "kernel_size":  (3,     integer),
        "weight":       (Auto,  auto|tensor),
        "bias":         (None,  none|auto|tensor),
        "stride":       (1,     integer|integer&integer|integer&integer&integer),
        "padding":      (0,     integer),
        "dilation":     (1,     integer),
        "groups":       (1,     integer),
        "activation":   ("id",  activation),
}

layers["Conv1d"]=Conv1d    
layers["Conv2d"]=Conv2d   
layers["Conv3d"]=Conv3d   
layers["ConvNd"]=ConvNd   
     
if __name__ == "__main__":
    print(ConvNd.args)
    print(layers)
    print([x.__name__ for x in layers.values()])
    fac = Layer_Factory(ConvNd)
    """fac.weight = torch.FloatTensor([[[
        [-1,0,1]
    ]]])"""
    fac.set_input_shape("input",None,1,6)
    fac.set_output_shape("output",None,5,None)
    fac.activation = "softshrink"
    fac.bias = Auto
    x = fac()
    print(x.weight)
    print(x.bias)
    pic = torch.rand(1,1,10)
    print(x.output_shape.output)
    print(x.input_shape.input)
    print(x(pic).output)
