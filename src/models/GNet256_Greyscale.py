import torch
import torch.nn as nn
import sys
sys.path.append('../utils/')
from utils import single_conv, conv_grid_search, conv_path_search

class OptGenGreyscale256(nn.Module):
    '''
    Generator Class for WGAN for greyscale images
    returns a value for the intensitity of the RGB channels
    for each pixel
    '''
    def __init__(self, ngpu, num_conv_layers, drop_conv2):
        super(OptGenGreyscale256, self).__init__()
        self.ngpu = ngpu
        self.drop_conv2 = drop_conv2
        self.num_filters = [3] 
        self.num_filters.extend([2**(i+5) for i in range(num_conv_layers-2)])
        self.num_filters.append(2**4) # starting levels -> total 4 x 4 x 16 = 256 params -> tried 2**9 channels, the model is too slow
        self.num_conv_layers = num_conv_layers
        self.strides = []
        self.paddings = []
        self.kernelSizes = []
        self.out_size = []
        if self.num_conv_layers == 3:
            # solution: {'ins': [256, 36.0, 12.0], 'outs': [36.0, 12.0, 4.0], 'kernel_sizes': [11, 3, 3], 'paddings': [0, 0, 0], 'strides': [7, 3, 3]}
            s3, c3 = conv_path_search(ins = 256, kernel_sizes = [11, 9, 7, 5, 3], 
                          strides = [1,3,5, 7], paddings = [0,1], convs = 3, out = 4, verbose = False)
            solution = s3[-1]
            self.strides = solution['strides']
            self.paddings = solution['paddings']
            self.kernelSizes = solution['kernel_sizes']
        if self.num_conv_layers == 4:
            # solution:{'ins': [256, 84.0, 26.0, 8.0], 'outs': [84.0, 26.0, 8.0, 4.0], 'kernel_sizes': [9, 9, 5, 5], 'paddings': [1, 0, 0, 0], 'strides': [3, 3, 3, 1]}
            s4, c4 = conv_path_search(ins = 256, kernel_sizes = [11, 9, 7, 5, 3], 
                          strides = [1,3,5, 7], paddings = [0,1], convs = 4, out = 4, verbose = False)
            solution = s4[-1] 
            self.strides = solution['strides']
            self.paddings = solution['paddings']
            self.kernelSizes = solution['kernel_sizes']
        # same scheme as for DNet, but inverted
        self.num_filters.reverse()
        self.strides.reverse()
        self.paddings.reverse()
        self.kernelSizes.reverse()
        
        self.main = nn.Sequential(
            # input is Z, going into a convolution with dimensions c=nz, h=1, w=1
            nn.ConvTranspose2d(in_channels=self.num_filters[0], #deconvolution!
                               out_channels=self.num_filters[0], #ngf * 8, 
                               kernel_size=self.kernelSizes[0], 
                               stride=self.strides[0], 
                               padding=self.paddings[0], 
                               bias=False), 
            nn.BatchNorm2d(self.num_filters[0]),
            nn.ReLU(True)
        )
        self.out_size.append([self.num_filters[0], (1-1)*self.strides[0]+self.kernelSizes[0]-2*self.paddings[0]])
        self.num_modules = 3
        for i in range(1, num_conv_layers):
            self.main.add_module(str(4*i-1)+"): TransConv_"+str(i+1), nn.ConvTranspose2d(in_channels=self.num_filters[i-1],
                                                            out_channels=self.num_filters[i],
                                                            kernel_size=self.kernelSizes[i],
                                                            stride=self.strides[i],
                                                            padding=self.paddings[i],
                                                            bias=False))
            self.out_size.append([self.num_filters[i], (self.out_size[i-1][1]-1)*self.strides[i]+self.kernelSizes[i]-2*self.paddings[i]])
            self.num_modules += 1
            if i + 1 < num_conv_layers:
                if self.drop_conv2 > 0:
                    self.main.add_module(str(4*i)+"): DropOut_" + str(i+1), nn.Dropout2d(p=self.drop_conv2))
                    self.num_modules += 1
                self.main.add_module(str(1+4*i)+"): BatchNorm_" + str(i+1), nn.BatchNorm2d(self.num_filters[i]))
                self.main.add_module(str(2+4*i)+"): ReLU_" + str(i+1), nn.ReLU(True))
                self.num_modules += 2
        # outputlayer as tanH, since it retuns the RGB value between -1 and 1
        self.main.add_module(str(self.num_modules), nn.Tanh()) # not nn.Tanh() # not sigmoid
        ## print(f"Progression of the sizes in the deconvolution: {self.out_size}")
    
    def forward(self, x):
        # starting with a pic 4x4 pixels with 2**4 channels
        x = x.view(-1, self.num_filters[0], 4, 4) 
        return self.main(x)