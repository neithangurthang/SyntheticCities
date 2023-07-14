import os
import sys
# if os.getcwd().split('/')[-1] == 'scripts':
#     os.chdir('../')
sys.path.append('../')
# print(os.getcwd())

from src.images.dataloader import CadastralImage, load_folder
from src.models.DCGAN import Generator, Discriminator
import shutil
import argparse
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

# Set random seed for reproducibility
manual_seed = 999
random.seed(manual_seed)
torch.manual_seed(manual_seed)
torch.use_deterministic_algorithms(False) # Needed for linear layers to perform correctly with GPU

# Root directory for dataset
dataroot = "../data/cadastralExportRGB/train/"

# Number of workers for dataloader
workers = 2

# Batch size during training
batch_size = 200
# Spatial size of training images. All images will be resized to this
#   size using a transformer.
image_size = 300

# Multiplyer for image re-sizing
# 11 if resolution is 64 x 64, 3.15 if resolution is 300 x 300
mult = 3.15

# Number of channels in the training images. For color images this is 3
nc = 3

# Size of z latent vector (i.e. size of generator input)
nz = 100

# Size of feature maps in generator
ngf = 64

# Size of feature maps in discriminator
ndf = 64

# Number of training epochs
num_epochs = 1000

# Learning rate for optimizers
lr = 1e-2

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1

# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

# Regularization for generator not being sure which color to put in a pixel
pixel_reg_rate = 1

# We can use an image folder dataset the way we have it setup
if os.path.exists(dataroot + '/.ipynb_checkpoints'):
    shutil.rmtree(dataroot + '/.ipynb_checkpoints')
    
# dataset = load_folder(dataroot, resolution=(image_size, image_size), mult=mult, device='cpu')

dataset = torch.load('../data/dataset.pkl')

dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
                                             shuffle=True, num_workers=4, drop_last=True)

# custom weights initialization called on ``netG`` and ``netD``
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
        
# Create the generator
netG = Generator(nz, nc, ngf).to(device)

# Handle multi-GPU if desired
if (device.type == 'cuda') and (ngpu > 1):
    netG = nn.DataParallel(netG, list(range(ngpu)))

# Apply the ``weights_init`` function to randomly initialize all weights
#  to ``mean=0``, ``stdev=0.02``.
netG.apply(weights_init)

# Create the Discriminator
netD = Discriminator(nc, ndf).to(device)

# Handle multi-GPU if desired
if (device.type == 'cuda') and (ngpu > 1):
    netD = nn.DataParallel(netD, list(range(ngpu)))
    
# Apply the ``weights_init`` function to randomly initialize all weights
# like this: ``to mean=0, stdev=0.2``.
netD.apply(weights_init)

# Initialize the ``BCELoss`` function
criterion = nn.BCELoss()

# Create batch of latent vectors that we will use to visualize
#  the progression of the generator
fixed_noise = torch.randn(64, nz, 1, 1, device=device)

# Establish convention for real and fake labels during training
real_label = 1.
fake_label = 0.

# Setup Adam optimizers for both G and D
optimizerD = optim.Adam(netD.parameters(), lr=lr)
optimizerG = optim.Adam(netG.parameters(), lr=lr)

# Training Loop

# Lists to keep track of progress
img_list = []
G_losses = []
D_losses = []
iters = 0

print("Starting Training Loop...")
# For each epoch
for epoch in range(num_epochs):
    # For each batch in the dataloader
    for i, data in enumerate(dataloader):
        
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        ## Train with all-real batch
        netD.zero_grad()
        # Format batch
        real_cpu = data[0].to(device)
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
        # Forward pass real batch through D
        output = netD(real_cpu).view(-1)
        # Calculate loss on all-real batch
        # errD_real = criterion(output, label)
        errD_real = -output.mean()
        # Calculate gradients for D in backward pass
        errD_real.backward()
        D_x = output.mean().item()

        ## Train with all-fake batch
        # Generate batch of latent vectors
        noise = torch.randn(b_size, nz, 1, 1, device=device)
        # Generate fake image batch with G
        fake = netG(noise)
        label.fill_(fake_label)
        # Classify all fake batch with D
        output = netD(fake.detach()).view(-1)
        # Calculate D's loss on the all-fake batch
        # errD_fake = criterion(output, label)
        errD_fake = output.mean()
        # Calculate the gradients for this batch, accumulated (summed) with previous gradients
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        # Compute error of D as sum over the fake and the real batches
        errD = errD_real + errD_fake
        # Update D
        optimizerD.step()
        for p in netD.parameters():
            p.data.clamp_(-0.05, 0.05)

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = netD(fake).view(-1)
        # Calculate G's loss based on this output
        # errG = criterion(output, label)
        # relu = nn.ReLU()
        pixel_reg = torch.mean(torch.sum(torch.log(fake + 1e-5), axis=1))
        errG = -output.mean() # + pixel_reg_rate * pixel_reg
        # Calculate gradients for G
        errG.backward()
        D_G_z2 = output.mean().item()
        # Update G
        optimizerG.step()
        
        # Output training stats
        if i % 50 == 0:
            print(f'[{epoch}/{num_epochs}][{i}/{len(dataloader)}] Loss_D: {errD.item():.4f} Loss_G: {errG.item():.4f} Pixel_Reg: {pixel_reg.item():.4f}')
        
        # Save Losses for plotting later
        G_losses.append(errG.item())
        D_losses.append(errD.item())
        
        # Check how the generator is doing by saving G's output on fixed_noise
        if (iters % 250 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()
                fake_grid = vutils.make_grid(fake, padding=2, normalize=True)
                vutils.save_image(fake_grid, f'../reports/fake_{iters}.png')
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
            
        iters += 1
    torch.save(netD, '../models/netD.pkl')
    torch.save(netG, '../models/netG.pkl')
