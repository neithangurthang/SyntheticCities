import torch
import torch.nn as nn
import sys
sys.path.append('../utils/')
from utils import single_conv, conv_grid_search, conv_path_search

ngpu = torch.cuda.device_count()
nz = 100

# Generator Class WGAN

class OptGen(nn.Module):
    def __init__(self, ngpu, num_conv_layers, drop_conv2):
        super(OptGen, self).__init__()
        self.ngpu = ngpu
        self.drop_conv2 = drop_conv2
        self.num_filters = [3] 
        self.num_filters.extend([2**(i+6) for i in range(num_conv_layers-1)])
        self.num_conv_layers = num_conv_layers
        self.strides = []
        self.paddings = []
        self.kernelSizes = []
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
        # same scheme as for DNet, but inverted
        self.num_filters.reverse()
        self.strides.reverse()
        self.paddings.reverse()
        self.kernelSizes.reverse()
        
        self.main = nn.Sequential(
            # input is Z, going into a convolution with dimensions c=nz, h=1, w=1
            # output size: (2**6) x 4 x 4 O=(I-1)*s+k-2p
            nn.ConvTranspose2d(in_channels=nz, #deconvolution!
                               out_channels=self.num_filters[0], #ngf * 8, 
                               kernel_size=self.kernelSizes[0], 
                               stride=self.strides[0], 
                               padding=self.paddings[0], 
                               bias=False), # (1-1)*1+4-2*0=4
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
                self.main.add_module(str(4*i)+"): DropOut_" + str(i+1), nn.Dropout2d(p=self.drop_conv2))
                self.main.add_module(str(1+4*i)+"): BatchNorm_" + str(i+1), nn.BatchNorm2d(self.num_filters[i]))
                self.main.add_module(str(2+4*i)+"): ReLU_" + str(i+1), nn.ReLU(True))
                self.num_modules += 3
        # softmax as the probability that a pixels belongs to a certain class
        self.main.add_module(str(self.num_modules), nn.Softmax(dim=1)) # not nn.Tanh() # not sigmoid
        ## print(f"Progression of the sizes in the deconvolution: {self.out_size}")
    
    def forward(self, x):
        return self.main(x)