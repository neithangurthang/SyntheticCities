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
from src.models.DCGAN import Generator, Discriminator
from src.utils.regularizers import gradient_penalty

# set random seed for reproducibility
manual_seed = 999
random.seed(manual_seed)
torch.manual_seed(manual_seed)
torch.use_deterministic_algorithms(False) # needed for linear layers to perform correctly with GPU

## CONSTANTS ##

DATAROOT = "../data/cadastralExportRGB/train/"  # root directory for dataset
G_PATH = "../models/netG_dcgan.pkl"
D_PATH = "../models/netD_dcgan.pkl"
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
    
# dataset = load_folder(DATAROOT, resolution=(IMG_SIZE, IMG_SIZE), mult=MULT, device='cpu')

test = torch.load('../data/test.pkl')
netG = torch.load(G_PATH)
netD = torch.load(D_PATH)