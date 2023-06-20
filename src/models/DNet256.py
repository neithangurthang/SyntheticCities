import torch
import torch.nn as nn
import sys
sys.path.append('../utils/')
from utils import single_conv, conv_grid_search, conv_path_search

class OptDis256(nn.Module):
    '''
    Returns a critic on the probability that an image 3x64x64 is real or fake
    high == real
    low == fake
    '''
    def __init__(self, ngpu, num_conv_layers, in_channels = 3):
        super(OptDis256, self).__init__()
        self.ngpu = ngpu
        self.num_filters = [2**(i+6) for i in range(num_conv_layers)]
        # self.num_filters.append(1)
        self.num_conv_layers = num_conv_layers
        self.strides = []
        self.paddings = []
        self.kernelSizes = []
        self.numberChannels = in_channels # could be an input
        self.out_size = []
        if self.num_conv_layers == 3:
            # solution: {'ins': [256, 36.0, 12.0], 'outs': [36.0, 12.0, 4.0], 'kernel_sizes': [11, 3, 3], 'paddings': [0, 0, 0], 'strides': [7, 3, 3]}
            s3, c3 = conv_path_search(ins = 256, kernel_sizes = [11, 9, 7, 5, 3], 
                          strides = [1,3,5, 7], paddings = [0,1], convs = 3, out = 4, verbose = False)
            solution = s3[-1]
            self.strides = solution['strides']
            self.paddings = solution['paddings']
            self.kernelSizes = solution['kernel_sizes']
            self.out_size = solution['outs']
        if self.num_conv_layers == 4:
            # solution:{'ins': [256, 84.0, 26.0, 8.0], 'outs': [84.0, 26.0, 8.0, 4.0], 'kernel_sizes': [9, 9, 5, 5], 'paddings': [1, 0, 0, 0], 'strides': [3, 3, 3, 1]}
            s4, c4 = conv_path_search(ins = 256, kernel_sizes = [11, 9, 7, 5, 3], 
                          strides = [1,3,5, 7], paddings = [0,1], convs = 4, out = 4, verbose = False)
            solution = s4[-1] 
            self.strides = solution['strides']
            self.paddings = solution['paddings']
            self.kernelSizes = solution['kernel_sizes']
            self.out_size = solution['outs']
            
        self.main = nn.Sequential(
            nn.Conv2d(in_channels=self.numberChannels, 
                      out_channels=self.num_filters[0], 
                      kernel_size=self.kernelSizes[0], 
                      stride=self.strides[0], 
                      padding=self.paddings[0], 
                      bias=False),
            nn.LeakyReLU(0.2, inplace=True))
        for i in range(1, num_conv_layers):
            self.main.add_module(str(3*i-1)+"): Conv_"+str(i+1), nn.Conv2d(in_channels=self.num_filters[i-1],
                                                            out_channels=self.num_filters[i],
                                                            kernel_size=self.kernelSizes[i],
                                                            stride=self.strides[i],
                                                            padding=self.paddings[i],
                                                            bias=False))
            if i + 1 < num_conv_layers: 
                self.main.add_module(str(3*i)+"): BatchNorm_" + str(i+1), nn.BatchNorm2d(self.num_filters[i]))
                self.main.add_module(str(1+3*i)+"): LeakyReLU_" + str(i+1), nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.main.add_module('Flatten', nn.Flatten())
        self.main.add_module('Fully Connected 1', nn.Linear(int(self.num_filters[-1]*self.out_size[-1]*self.out_size[-1]), 2**9))
        self.main.add_module('ReLU', nn.ReLU(True))
        self.main.add_module('Fully Connected 2', nn.Linear(in_features = 2**9, out_features = 1, bias = True))
        # NO ACTIVATION FUNCTION AT THE END: the idea is that the output domain for D is richer and can give a richer critict
        # avoiding local minima for G
        
        # self.main.add_module(str(3*i)+"): Sigmoid", nn.Sigmoid()) 
        # self.main.add_module(str(3*i)+"): Tanh", nn.tanh()) #  or nothing
    
    def forward(self, input):
        return self.main(input)