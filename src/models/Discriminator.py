import torch
import torch.nn as nn
import sys
sys.path.append('../utils/')
from utils import single_conv, conv_grid_search, conv_path_search

# Discriminator Class WGAN

class OptDis(nn.Module):
    '''
    Returns a critic on the probability that an image 3x64x64 is real or fake
    high == real
    low == fake
    '''
    def __init__(self, ngpu, num_conv_layers):
        super(OptDis, self).__init__()
        self.ngpu = ngpu
        self.num_filters = [2**(i+6) for i in range(num_conv_layers-1)]
        self.num_filters.append(1)
        self.num_conv_layers = num_conv_layers
        self.strides = []
        self.paddings = []
        self.kernelSizes = []
        self.numberChannels = 3 # could be an input
        self.out_size = []
        if self.num_conv_layers == 3:
            # solution 3: {'ins': [64, 22.0, 7.0], 'outs': [22.0, 7.0, 1.0], 'kernel_sizes': [3, 4, 7], 'paddings': [1, 0, 0], 'strides': [3, 3, 3]}
            s3, c3 = conv_path_search(64, kernel_sizes = [3,4,5,7], 
                                      strides = list(range(1,4)), paddings = [0,1], convs = 3)
            solution = s3[-1]
            self.strides = solution['strides']
            self.paddings = solution['paddings']
            self.kernelSizes = solution['kernel_sizes']
        if self.num_conv_layers == 4:
            # solution 2: {'ins': [64, 31.0, 14.0, 4.0], 'outs': [31.0, 14.0, 4.0, 1.0], 'kernel_sizes': [4, 5, 5, 4], 'paddings': [0, 0, 0, 0], 'strides': [2, 2, 3, 3]}
            s4, c4 = conv_path_search(64, kernel_sizes = [3,4,5], 
                                      strides = list(range(2,4)), paddings = [0], convs = 4)
            solution = s4[-1] 
            self.strides = solution['strides']
            self.paddings = solution['paddings']
            self.kernelSizes = solution['kernel_sizes']
            
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64 -> output nc = 2**6 x 32 x 32
            nn.Conv2d(in_channels=self.numberChannels, 
                      out_channels=self.num_filters[0], 
                      kernel_size=self.kernelSizes[0], 
                      stride=self.strides[0], 
                      padding=self.paddings[0], 
                      bias=False), # (64-4+2)/2 + 1=32
            nn.LeakyReLU(0.2, inplace=True))
        for i in range(1, num_conv_layers):
            # input is nc=(2**(i+5) x 32 x 32 -> output nc = 2**(i+6) x 32*2**(-i) x 32*2**(-i)
            self.main.add_module(str(3*i-1)+"): Conv_"+str(i+1), nn.Conv2d(in_channels=self.num_filters[i-1],
                                                            out_channels=self.num_filters[i],
                                                            kernel_size=self.kernelSizes[i],
                                                            stride=self.strides[i],
                                                            padding=self.paddings[i],
                                                            bias=False))
            if i + 1 < num_conv_layers: 
                self.main.add_module(str(3*i)+"): BatchNorm_" + str(i+1), nn.BatchNorm2d(self.num_filters[i]))
                self.main.add_module(str(1+3*i)+"): LeakyReLU_" + str(i+1), nn.LeakyReLU(negative_slope=0.2, inplace=True))
        # NO ACTIVATION FUNCTION AT THE END: the idea is that the output domain for D is richer and can give a richer critict
        # avoiding local minima for G
        
        # self.main.add_module(str(3*i)+"): Sigmoid", nn.Sigmoid()) 
        # self.main.add_module(str(3*i)+"): Tanh", nn.tanh()) #  or nothing
    
    def forward(self, input):
        return self.main(input)