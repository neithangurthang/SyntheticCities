import torch
import torch.nn as nn
import sys
sys.path.append('../utils/')
from utils import single_conv, conv_grid_search, conv_path_search

# Coverts conditions into feature vectors
class ConditionNetG(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 64, alpha: float = 0.2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # From label feature: 3 => 64
        self.fc = nn.Sequential(
            nn.Linear(self.in_channels, 64),
            nn.BatchNorm1d(64),
            nn.Linear(64, self.out_channels),
            nn.LeakyReLU(alpha))
        
    def forward(self, labels: torch.Tensor):
        return self.fc(labels).view(-1,self.out_channels,1,1)


# Generator Class
class CondGen(nn.Module):
    def __init__(self, nz:int, num_conv_layers: int, drop_conv2: float, labels_numb: int, 
                 noise_filters = 1, noise_pixels = 8):
        """
        This is the generator class for conditional GAN
        The input params are: 
            nz: dimension of the input noise z
            num_conv_layers: number of convolutional layers, between 3 and 6
            drop_conv2: probabiltity of the dropout for the dropout layer
            labels_numb: number of labels for the conditional distributions
        The output is a batch of images with 3 channels and resolution 64x64
        
        """
        super(CondGen, self).__init__()
        self.ngpu = torch.cuda.device_count()
        self.device = torch.device("cuda:0" if (torch.cuda.is_available() and self.ngpu > 0) else "cpu")
        self.drop_conv2 = drop_conv2
        self.nz = nz
        self.labels_numb = labels_numb
        self.num_filters = [1] 
        self.num_filters.extend([2**(i+4) for i in range(num_conv_layers-2)])
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
            solution = s3[-1]
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
        
        ### Generator Architecture
        self.main = nn.Sequential(
            nn.ConvTranspose2d(in_channels=2, #deconvolution!
                               out_channels=self.num_filters[0], #ngf * 8, 
                               kernel_size=self.kernelSizes[0], 
                               stride=self.strides[0], 
                               padding=self.paddings[0], 
                               bias=False), # (1-1)*1+4-2*0=4
            nn.BatchNorm2d(self.num_filters[0]),
            nn.ReLU(True)
        )
        self.num_modules = 3
        for i in range(1, self.num_conv_layers):
            self.main.add_module(str(4*i-1)+": TransConv_"+str(i+1), nn.ConvTranspose2d(in_channels=self.num_filters[i-1],
                                                            out_channels=self.num_filters[i],
                                                            kernel_size=self.kernelSizes[i],
                                                            stride=self.strides[i],
                                                            padding=self.paddings[i],
                                                            bias=False))

            self.num_modules += 1
            if i + 1 < self.num_conv_layers:
                self.main.add_module(str(4*i)+": DropOut_" + str(i+1), nn.Dropout2d(p=self.drop_conv2))
                self.main.add_module(str(1+4*i)+": BatchNorm_" + str(i+1), nn.BatchNorm2d(self.num_filters[i]))
                self.main.add_module(str(2+4*i)+": ReLU_" + str(i+1), nn.ReLU(True))
                self.num_modules += 3
            
        self.main.add_module(str(self.num_modules), nn.Tanh())
    
        self.cond = nn.Sequential(ConditionNetG())
    
    def forward(self, z: torch.Tensor, labels: torch.Tensor):
        '''here z and labels are combined in the input for main'''
        z = z.view(-1, self.noise_filters, self.noise_pixels, self.noise_pixels)
        c = self.cond(labels).view(-1, self.noise_filters, self.noise_pixels, self.noise_pixels)

        x = torch.cat([z, c], 1)         
        x = self.main(x)
        return x