import torch.multiprocessing as mp

import shutil
import argparse
import random
import sys
sys.path.append('../')
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.utils as vutils

from src.images.dataloader import CadastralImage, load_folder
from src.models.DCGAN_BCE import Generator, Discriminator
from src.utils.regularizers import gradient_penalty

# set random seed for reproducibility
manual_seed = 999
random.seed(manual_seed)
torch.manual_seed(manual_seed)
torch.use_deterministic_algorithms(False) # needed for linear layers to perform correctly with GPU

## CONSTANTS ##

DATAROOT = "../data/cadastralExportRGB/train/"  # root directory for dataset
WORKERS = 4  # number of workers for dataloader
BATCH_SIZE = 128  # batch size during training
IMG_SIZE = 300  # spatial size of training images (to be resized to)
MULT = 3.15  # re-size factor: 11 if resolution is 64 x 64, 3.15 if resolution is 300 x 300
NC = 3  # number of entities aka channels in the training images
NZ = 100  # size of noise vector (i.e. size of generator input)
NGF = 32  # base size of feature maps in generator
NDF = 32  # base size of feature maps in discriminator
NUM_EPOCHS = 1000  # number of training epochs
LR = 1e-4  # learning rate for both optimizers
DEVICE = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")  # which device to run on
GP_RATE = 1  # gradient penalty rate
COND_RATE = 1  # conditional regularization (based on percentage of Roads - Greens - Buildings)
DROP_RATE = 0.05  # dropout rate for generator
MAX_FRAC = 0.15  # threshold of maximum fraction of particular condition
MIN_FRAC = 0.01
STD_RATE = 1  # regularization on having not so wide distribution

# We can use an image folder dataset the way we have it setup
if os.path.exists(DATAROOT + '/.ipynb_checkpoints'):
    shutil.rmtree(DATAROOT + '/.ipynb_checkpoints')

train = torch.load('../data/train.pkl')

dataloader = torch.utils.data.DataLoader(train, batch_size=BATCH_SIZE, 
                                             shuffle=True, num_workers=WORKERS, drop_last=True)

# custom weights initialization called on ``netG`` and ``netD``
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
        
# Create the generator
netG = Generator(NZ, NC, NGF, DROP_RATE).to(DEVICE)
# netG = Generator(NZ + NC, NC, NGF).to(DEVICE)  # for the case of conditions
netG.apply(weights_init)  # initialize weights for generator

# Create the Discriminator
netD = Discriminator(NC, NDF).to(DEVICE)
netD.apply(weights_init)  # initialize weights for discriminator

fixed_noise = torch.randn(64, NZ, 1, 1, device=DEVICE)

# setup Adam optimizers for both G and D
optimizerD = optim.Adam(netD.parameters(), lr=LR)
optimizerG = optim.Adam(netG.parameters(), lr=LR)

criterion = nn.BCELoss()

# establish convention for real and fake labels during training
real_label = 1.
fake_label = 0.

# Training Loop

# Lists to keep track of progress
img_list = []
G_losses = []
D_losses = []
iters = 0


def train():
    print("Starting Training Loop...")
    for epoch in range(NUM_EPOCHS):
        for i, data in enumerate(dataloader):
            
            ############################
            # (1) Update D network: maximize D(x) - D(G(z))
            ###########################
            ## Train with all-real batch
            netD.zero_grad()
            # Format batch
            real_batch = data[0].to(DEVICE)
            # Extracting the conditions from real batch to fake them then
            # real_conds = torch.sum(real_batch, dim=(2, 3)) / (IMG_SIZE ** 2)
            # Filtering based on the conditions
            # indices = ((real_conds.max(dim=1).values > MAX_FRAC) == False).nonzero(as_tuple=True)[0]
            # Slicing and re-creting conditions based on filtered batch
            # real_batch = torch.index_select(real_batch, 0, indices)
            # real_conds = torch.index_select(real_conds, 0, indices)
            b_size = real_batch.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=DEVICE)
            # Forward pass real batch through D and make gradient update
            output = netD(real_batch).view(-1)
            errD_real = criterion(output, label)
            errD_real.backward()

            ## Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, NZ, 1, 1, device=DEVICE)
            label.fill_(fake_label)
            # real_conds = real_conds.view(real_conds.shape[0], real_conds.shape[1], 1, 1)
            # noise = torch.cat((noise, real_conds), dim=1)
            # Generate fake image batch with G
            fake = netG(noise)
            # fake_conds = torch.sum(fake, dim=(2, 3)) / (IMG_SIZE ** 2)
            # Conditional regularization
            # cond_reg = torch.mean((fake_conds - noise[:, -NC:, :, :]) ** 2)
            # Classify all fake batch with D
            output = netD(fake.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, label)
            # Calculate the gradients for this batch, accumulated (summed) with previous gradients
            errD_fake.backward()
            # Compute error of D as sum over the fake and the real batches
            # gp = gradient_penalty(netD, real_batch, fake, DEVICE) # lambda gradient penalty = 0.2
            errD = errD_real + errD_fake # + STD_RATE * output.std()  # + GP_RATE * gp
            # Update D
            optimizerD.step()
            # Introduce discriminator gradient clipping so that it doesn't learn too fast
            for p in netD.parameters():
                p.data.clamp_(-0.05, 0.05)

            ############################
            # (2) Update G network: maximize D(G(z))
            ###########################
            netG.train()
            netG.zero_grad()
            label.fill_(real_label)
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = netD(fake).view(-1)
            # Calculate G's loss based on this output with gradient update
            errG = criterion(output, label) # + STD_RATE * output.std() # + COND_RATE * cond_reg
            errG.backward()
            optimizerG.step()
            
            # Output training stats
            if i == 0:
                print(f'[{epoch}/{NUM_EPOCHS}][{i}/{len(dataloader)}] Loss_D: {errD.item():.4f} Loss_G: {errG.item():.4f}')  # Gradient penalty: {gp.item():.4f} Conditional Reg: {COND_RATE * cond_reg.item():.4f}
            
            # Save Losses for plotting later
            # G_losses.append(errG.item())
            # D_losses.append(errD.item())
            
            # Check how the generator is doing by saving G's output on fixed_noise
            if (i == 0) or ((epoch == NUM_EPOCHS-1) and (i == len(dataloader)-1)):
                with torch.no_grad():
                    netG.eval()
                    fake = netG(fixed_noise).detach().cpu()
                    fake_grid = vutils.make_grid(fake, padding=2, normalize=True)
                    vutils.save_image(fake_grid, f'../reports/fake_{iters}.png')
                # img_list.append(fake_grid)
                
            iters += 1
        torch.save(netD, '../models/netD_Multiprocessing.pkl')
        torch.save(netG, '../models/netG_Multiprocessing.pkl')



##########################################################################

# START TRAINING PROCESS

##########################################################################

if __name__ == '__main__':
    num_processes = 4
    # model = MyModel()
    # NOTE: this is required for the ``fork`` method to work
    netG.share_memory()
    netD.share_memory()
    processes = []
    for rank in range(num_processes):
        p = mp.Process(target=train) # s, args=(model,))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()