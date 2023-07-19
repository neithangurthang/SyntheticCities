import torch
from torch import nn


class Generator(nn.Module):
    def __init__(self, nz, nc, ngf, drop_rate):
        super(Generator, self).__init__()
        self.softmax = nn.Softmax(dim=1)
        self.main = nn.Sequential(
            nn.ConvTranspose2d(in_channels=nz, out_channels=ngf * 8, kernel_size=2, stride=1, 
                               padding=0, bias=False),
            nn.Dropout2d(p=drop_rate),
            nn.BatchNorm2d(ngf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(in_channels=ngf * 8, out_channels=ngf * 4, kernel_size=4, stride=2, 
                               padding=0, bias=False),
            nn.Dropout2d(p=drop_rate),
            nn.BatchNorm2d(ngf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(in_channels=ngf * 4, out_channels=ngf * 2, kernel_size=5, stride=3, 
                               padding=1, bias=False),
            nn.Dropout2d(p=drop_rate),
            nn.BatchNorm2d(ngf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(in_channels=ngf * 2, out_channels=ngf, kernel_size=6, stride=4, 
                               padding=0, bias=False),
            nn.Dropout2d(p=drop_rate),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(in_channels=ngf, out_channels=nc, kernel_size=12, stride=4, 
                               padding=2, bias=False)
        )
    
    def forward(self, input_):
        input_ = self.main(input_)
        return self.softmax(input_)


# class Discriminator(nn.Module):
#     def __init__(self, nc, ndf):
#         super(Discriminator, self).__init__()
#         self.main = nn.Sequential(
#             nn.Conv2d(in_channels=nc, out_channels=ndf, kernel_size=23, stride=2, padding=0, 
#                       bias=False),
#             nn.BatchNorm2d(ndf),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(in_channels=ndf, out_channels=ndf * 2, kernel_size=15, stride=2, padding=0, 
#                       bias=False),
#             nn.BatchNorm2d(ndf * 2),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(in_channels=ndf * 2, out_channels=ndf * 4, kernel_size=9, stride=1, padding=0, 
#                       bias=False),
#             nn.BatchNorm2d(ndf * 4),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(in_channels=ndf * 4, out_channels=ndf * 8, kernel_size=5, stride=1, padding=0, 
#                       bias=False),
#             nn.BatchNorm2d(ndf * 8),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(in_channels=ndf * 8, out_channels=1, kernel_size=3, stride=1, padding=0, 
#                       bias=False),
#             nn.Tanhshrink()
#         )

#     def forward(self, input_):
#         return self.main(input_)


# class Discriminator(nn.Module):
#     def __init__(self, nc, ndf):
#         super(Discriminator, self).__init__()
#         self.main = nn.Sequential(
#             nn.Conv2d(in_channels=nc, out_channels=ndf, kernel_size=3, stride=1, padding=0, 
#                       bias=False),
#             nn.BatchNorm2d(ndf),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(in_channels=ndf, out_channels=ndf * 2, kernel_size=3, stride=2, padding=1, 
#                       bias=False),
#             nn.BatchNorm2d(ndf * 2),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(in_channels=ndf * 2, out_channels=ndf * 4, kernel_size=3, stride=2, padding=1, 
#                       bias=False),
#             nn.BatchNorm2d(ndf * 4),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(in_channels=ndf * 4, out_channels=ndf * 8, kernel_size=3, stride=2, padding=1, 
#                       bias=False),
#             nn.BatchNorm2d(ndf * 8),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(in_channels=ndf * 8, out_channels=1, kernel_size=3, stride=1, padding=0, 
#                       bias=False),
#             # nn.Tanhshrink()
#         )

#     def forward(self, input_):
#         return self.main(input_)
    
class Discriminator(nn.Module):
    def __init__(self, nc, ndf):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels=nc, out_channels=ndf, kernel_size=5, stride=2, padding=1, 
                      bias=False),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=ndf, out_channels=ndf * 2, kernel_size=5, stride=2, padding=1, 
                      bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=ndf * 2, out_channels=ndf * 4, kernel_size=5, stride=2, padding=1, 
                      bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=ndf * 4, out_channels=ndf * 8, kernel_size=3, stride=1, padding=0, 
                      bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=ndf * 8, out_channels=1, kernel_size=3, stride=1, padding=0, 
                      bias=False),
            nn.Tanhshrink()
        )

    def forward(self, input_):
        return self.main(input_)


# class Discriminator(nn.Module):
#     def __init__(self, nc, ndf):
#         super(Discriminator, self).__init__()
#         self.main = nn.Sequential(
#             nn.Conv2d(in_channels=nc, out_channels=ndf, kernel_size=11, stride=2, padding=1, 
#                       bias=False),
#             nn.BatchNorm2d(ndf),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(in_channels=ndf, out_channels=ndf * 2, kernel_size=9, stride=2, padding=1, 
#                       bias=False),
#             nn.BatchNorm2d(ndf * 2),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(in_channels=ndf * 2, out_channels=ndf * 4, kernel_size=7, stride=2, padding=1, 
#                       bias=False),
#             nn.BatchNorm2d(ndf * 4),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(in_channels=ndf * 4, out_channels=ndf * 8, kernel_size=5, stride=1, padding=0, 
#                       bias=False),
#             nn.BatchNorm2d(ndf * 8),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(in_channels=ndf * 8, out_channels=1, kernel_size=3, stride=1, padding=0, 
#                       bias=False)
#             # nn.Tanhshrink()
#         )

#     def forward(self, input_):
#         return self.main(input_)
