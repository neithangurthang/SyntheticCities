import torch
from torch import nn


class Generator(nn.Module):
    def __init__(self, nz, nc, ngf,ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.nz = nz
        self.nc = nc
        self.ngf = ngf
        self.main = nn.Sequential(
            nn.ConvTranspose2d(in_channels=self.nz, out_channels=ngf * 8, kernel_size=2, stride=1, 
                               padding=0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(in_channels=ngf * 8, out_channels=ngf * 4, kernel_size=4, stride=2, 
                               padding=0, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(in_channels=ngf * 4, out_channels=ngf * 2, kernel_size=5, stride=3, 
                               padding=1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(in_channels=ngf * 2, out_channels=ngf, kernel_size=6, stride=4, 
                               padding=0, bias=False),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(in_channels=ngf, out_channels=nc, kernel_size=12, stride=4, 
                               padding=2, bias=False),
            nn.Softmax(dim=1)
        )

    def forward(self, input_):
        return self.main(input_)
    
    
class Discriminator(nn.Module):
    def __init__(self, ndf, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.conv3d = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=ndf, kernel_size=3, stride=2, padding=0, 
                      bias=False),
            nn.BatchNorm3d(ndf),
            nn.LeakyReLU(0.2, inplace=True))
        self.main = nn.Sequential(
            nn.Conv2d(in_channels=ndf, out_channels=ndf * 2, kernel_size=3, stride=2, padding=1, 
                      bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=ndf * 2, out_channels=ndf * 4, kernel_size=3, stride=2, padding=1, 
                      bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=ndf * 4, out_channels=ndf * 8, kernel_size=3, stride=2, padding=1, 
                      bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=ndf * 8, out_channels=ndf * 8, kernel_size=3, stride=1, padding=0, 
                      bias=False),
            nn.Flatten(),
            nn.Linear(int(ndf * 8 * 289), 512),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Linear(512, 32),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Linear(32, 1),
            # nn.Sigmoid()
            nn.Tanhshrink()
        )

    def forward(self, input_, batch_size, nc, image_size):
        input_ = input_.reshape(batch_size, 1, nc, image_size, image_size)  # required for Conv3d
        input_ = self.conv3d(input_)
        input_ = torch.squeeze(input_)
        return self.main(input_)
