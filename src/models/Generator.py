import torch
import torch.nn as nn

ngpu = torch.cuda.device_count()
nz = 100

# 5. Generator Class
class OptGen(nn.Module):
    def __init__(self, ngpu, num_conv_layers):
        super(OptGen, self).__init__()
        self.ngpu = ngpu
        self.num_filters = [3] 
        self.num_filters.extend([2**(i+6) for i in range(num_conv_layers-1)])
        self.strides = [2]
        self.paddings = [1]
        self.kernelSizes = [4]
        self.out_size = []
        if num_conv_layers == 3:
            self.strides.extend([2,2])
            self.paddings.extend([0,0])
            self.kernelSizes.extend([14,10])
        if num_conv_layers == 4:
            self.strides.extend([2,2,2])
            self.paddings.extend([1,0,0])
            self.kernelSizes.extend([4,6,6])
        if num_conv_layers == 5:
            self.strides.extend([2,2,2,2])
            self.paddings.extend([1,1,1,0])
            self.kernelSizes.extend([4,4,4,4])
        if num_conv_layers == 6:
            self.strides.extend([2,2,2,2,2])
            self.paddings.extend([1,1,1,1,1])
            self.kernelSizes.extend([4,4,4,4,4])
        # self.out_size = []
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
            self.main.add_module(str(3*i-1)+"): TransConv_"+str(i+1), nn.ConvTranspose2d(in_channels=self.num_filters[i-1],
                                                            out_channels=self.num_filters[i],
                                                            kernel_size=self.kernelSizes[i],
                                                            stride=self.strides[i],
                                                            padding=self.paddings[i],
                                                            bias=False))
            self.out_size.append([self.num_filters[i], (self.out_size[i-1][1]-1)*self.strides[i]+self.kernelSizes[i]-2*self.paddings[i]])
            self.num_modules += 1
            if i + 1 < num_conv_layers: 
                self.main.add_module(str(3*i)+"): BatchNorm_" + str(i+1), nn.BatchNorm2d(self.num_filters[i]))
                self.main.add_module(str(1+3*i)+"): ReLU_" + str(i+1), nn.ReLU(True))
                self.num_modules += 2
            
        self.main.add_module(str(self.num_modules), nn.Tanh())
        ## print(f"Progression of the sizes in the deconvolution: {self.out_size}")
    
    def forward(self, input):
        return self.main(input)