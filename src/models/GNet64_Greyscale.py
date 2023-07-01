import torch
import torch.nn as nn
import sys
sys.path.append('../utils/')
from utils import single_conv, conv_grid_search, conv_path_search


class OptGenGreyscale(nn.Module):
    '''
    Generator Class for WGAN for greyscale images
    returns a value for the intensitity of the RGB channels
    for each pixel
    '''
    def __init__(self, ngpu, num_conv_layers, drop_conv2, noise_filters = 1, noise_pixels = 8):
        super(OptGenGreyscale, self).__init__()
        self.ngpu = ngpu
        self.drop_conv2 = drop_conv2
        self.num_filters = [1] 
        self.num_filters.extend([2**(i+6) for i in range(num_conv_layers-1)])
        self.num_filters.append(noise_filters)
        self.num_conv_layers = num_conv_layers
        self.strides = []
        self.paddings = []
        self.kernelSizes = []
        self.out_size = []
        self.noise_filters = noise_filters
        self.noise_pixels = noise_pixels
        if self.num_conv_layers == 3:
            # solution: {'ins': [64, 22.0, 8.0], 'outs': [22.0, 8.0, 8.0], 'kernel_sizes': [3, 3, 1], 'paddings': [1, 1, 0], 'strides': [3, 3, 1]}
            s3, c3 = conv_path_search(ins = 64, kernel_sizes = [7,5,3],
                                      strides = [5,3,1], paddings = [0,1], convs = 3, out = self.noise_pixels, 
                                      verbose = False)
            self.strides = solution['strides']
            self.paddings = solution['paddings']
            self.kernelSizes = solution['kernel_sizes']
            self.out_size = [int(i) for i in solution['ins']]
        if self.num_conv_layers == 4:
            # solution: {'ins': [64, 22.0, 8.0, 8.0], 'outs': [22.0, 8.0, 8.0, 8.0], 'kernel_sizes': [3, 3, 3, 3], 'paddings': [1, 1, 1, 1], 'strides': [3, 3, 1, 1]}
            s4, c4 = conv_path_search(ins = 64, kernel_sizes = [7, 5, 3], 
                                      strides = [5,3,1], paddings = [0,1], convs = 4, out = self.noise_pixels, 
                                      verbose = False)
            solution = s4[-1] 
            self.strides = solution['strides']
            self.paddings = solution['paddings']
            self.kernelSizes = solution['kernel_sizes']
            self.out_size = [int(i) for i in solution['ins']]
        # same scheme as for DNet, but inverted
        self.num_filters.reverse()
        self.strides.reverse()
        self.paddings.reverse()
        self.kernelSizes.reverse()
        self.out_size.reverse()
        
        # MAIN
        self.main = nn.Sequential(
            # input is Z, going into a convolution with dimensions c=nz, h=1, w=1
            nn.ConvTranspose2d(in_channels=self.num_filters[0], #deconvolution!
                               out_channels=self.num_filters[1], #ngf * 8, 
                               kernel_size=self.kernelSizes[0], 
                               stride=self.strides[0], 
                               padding=self.paddings[0], 
                               bias=False), 
            nn.BatchNorm2d(self.num_filters[1]),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.out_size.append([self.num_filters[0], (1-1)*self.strides[0]+self.kernelSizes[0]-2*self.paddings[0]])
        self.num_modules = 3
        for i in range(1, num_conv_layers):
            self.main.add_module(str(4*i-1)+"): TransConv_"+str(i+1), nn.ConvTranspose2d(in_channels=self.num_filters[i],
                                                            out_channels=self.num_filters[i+1],
                                                            kernel_size=self.kernelSizes[i],
                                                            stride=self.strides[i],
                                                            padding=self.paddings[i],
                                                            bias=False))
            # self.out_size.append([self.num_filters[i], (self.out_size[i-1][1]-1)*self.strides[i]+self.kernelSizes[i]-2*self.paddings[i]])
            self.num_modules += 1
            if i + 1 < num_conv_layers:
                if self.drop_conv2 > 0:
                    self.main.add_module(str(4*i)+"): DropOut_" + str(i+1), nn.Dropout2d(p=self.drop_conv2))
                    self.num_modules += 1
                self.main.add_module(str(1+4*i)+"): BatchNorm_" + str(i+1), nn.BatchNorm2d(self.num_filters[i+1]))
                self.main.add_module(str(2+4*i)+"): LeakyReLU_" + str(i+1), nn.LeakyReLU(negative_slope=0.2, inplace=True))
                self.num_modules += 2
        # outputlayer as tanH, since it retuns the RGB value between -1 and 1
        self.main.add_module(str(self.num_modules), nn.Tanh()) # not nn.Tanh() # not sigmoid
        ## print(f"Progression of the sizes in the deconvolution: {self.out_size}")
    
    def forward(self, x: torch.tensor):
        x = x.view(-1, self.noise_filters, self.noise_pixels, self.noise_pixels)
        return self.main(x)
    

