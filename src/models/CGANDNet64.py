import torch
import torch.nn as nn
import sys
sys.path.append('../utils/')
from utils import single_conv, conv_grid_search, conv_path_search
from OptimisationCGAN import getGreyscaleLabels

# Discriminator Class CGAN

class Condition(nn.Module):
    def __init__(self, num_conv_layers: int = 3, alpha: float = 0.2):
        super().__init__()
        self.channels = 2**(num_conv_layers+6)
        
        self.fc = nn.Sequential(
            nn.Linear(3, self.channels),
            nn.BatchNorm1d(self.channels),
            nn.Linear(self.channels, 8*8),
            nn.LeakyReLU(alpha))
        
    def forward(self, labels: torch.Tensor):
        return self.fc(labels).view(labels.size(0),-1,8,8)


class CondDis(nn.Module):
    def __init__(self, num_conv_layers:int):
        """
        This is the discriminator class for the conditional discriminator
        Params
        num_conv_layers: number of convolutional layers, domain from 3 to 6
        labels: list of 3 float with the frequency of each entity building, roads, other 
        The output is a tensor with a critic on each image
        
        """
        super(CondDis, self).__init__()
        self.ngpu = torch.cuda.device_count()
        self.device = torch.device("cuda:0" if (torch.cuda.is_available() and self.ngpu > 0) else "cpu")
        self.num_filters = [2**(i+6) for i in range(num_conv_layers-1)]
        self.num_filters.append(1)
        self.num_conv_layers = num_conv_layers
        self.strides = []
        self.paddings = []
        self.kernelSizes = []
        self.numberChannels = 1 # could be an input, 3 are the RGB channels
        self.out_size = []
        self.input_size = 64 # resolution of the image size
        self. num_conv_layers = num_conv_layers
        if self.num_conv_layers == 3:
            # solution: {'ins': [64, 22.0, 8.0], 'outs': [22.0, 8.0, 8.0], 'kernel_sizes': [3, 3, 3], 'paddings': [1, 1, 1], 'strides': [3, 3, 1]}
            s3, c3 = conv_path_search(ins = 64, kernel_sizes = [7,5,3],
                              strides = [5,3,1], paddings = [0,1], convs = 3, out = 8, verbose = False)
            solution = s3[-1]
            self.strides = solution['strides']
            self.paddings = solution['paddings']
            self.kernelSizes = solution['kernel_sizes']
        if self.num_conv_layers == 4:
            # solution: {'ins': [64, 22.0, 8.0, 8.0], 'outs': [22.0, 8.0, 8.0, 6.0], 'kernel_sizes': [3, 3, 3, 3], 'paddings': [1, 1, 1, 0], 'strides': [3, 3, 1, 1]}
            s4, c4 = conv_path_search(ins = 64, kernel_sizes = [7, 5, 3], 
                              strides = [5,3,1], paddings = [0,1], convs = 4, out = 8, verbose = False)
            solution = s4[-1]
            self.strides = solution['strides']
            self.paddings = solution['paddings']
            self.kernelSizes = solution['kernel_sizes']

        self.main = nn.Sequential(
            nn.Conv2d(in_channels=self.numberChannels, 
                      out_channels=self.num_filters[0], 
                      kernel_size=self.kernelSizes[0], 
                      stride=self.strides[0], 
                      padding=self.paddings[0], 
                      bias=False), # (64-4+2)/2 + 1=32
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
        
        self.cond = nn.Sequential(
            Condition(num_conv_layers=self.num_conv_layers)
        )
        
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.num_filters[0], self.num_filters[0]),
            nn.BatchNorm1d(self.num_filters[0]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(self.num_filters[0], 1)
        )
        
    def forward(self, x):
        with torch.no_grad():
            real_labels = torch.stack([getGreyscaleLabels(x_i) for i, x_i in enumerate(torch.unbind(x, dim=0))], dim=0).view(-1,3)
        c = self.cond(real_labels.to(self.device))
        x = self.main(x)
        x = self.fc(x + c)
        return x