"""
this code optimises G and D on cadastral images

"""

##########################
#                        #
# 0 - Import Libraries   #
#                        #
##########################

import sys
import os
import random
import numpy as np
import datetime
import shutil
import logging

import torch
import torch.nn as nn
from torch import autograd
from tqdm.auto import tqdm
from torchvision import transforms
# from torchvision.datasets import MNIST
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
import torchvision.datasets as dset # all datasets available in torch vision. not sure if needed here
import torchvision.utils as vutils # draw bounding box, segmantation mask, keypoints. convert to rgb, make grid, save_image
import torch.optim as optim # optimizer for example optim.Adam([var1, var2], lr=0.0001)

import optuna
from optuna.trial import TrialState
import mlflow

import matplotlib.pyplot as plt
torch.manual_seed(0) # Set for testing purposes, please do not change!
import matplotlib.animation as animation
from IPython.display import HTML # to embed html in the Ipython output

sys.path.append("../../src/models/")
sys.path.append("../../src/utils/")

from Generator import OptGen
from Discriminator import OptDis
from utils import normalizeRGB
from utils import weights_init
from OptimisationFunctions import suggest_hyperparameters, trainModel, test, objective

#######################
#                     #
# 1 - Define Params   #
#                     #
#######################

dataroot = "../../../cadastralExportRGB" # Root directory for train dataset
workers = 2 # Number of workers for dataloader
batch_size = 64 # Batch size during training
image_size = 64 # Spatial size of training images. All images will be resized to this
nc = 3 # Number of channels in the training images. For color images this is 3
nz = 100 # Size of z latent vector (i.e. size of generator input)
ngf = 64 # Size of feature maps in generator
ndf = 64 # Size of feature maps in discriminator
num_epochs = 1000 # Number of training epochs
trials = 10 # number of trials
lr = 0.0002 # Learning rate for optimizers
beta1 = 0.5 # Beta1 hyperparam for Adam optimizers
lambda_gradient_penality = 0.2 # to adjust the Wasserstein distance with interpolation between real and fake data
ngpu = torch.cuda.device_count() # Number of GPUs available. Use 0 for CPU mode.
# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

#####################
#                   #
# 2 - LOAD DATASET  #
#                   #
#####################
        
# first delete the unnecessary folder 
if os.path.exists(dataroot + '/.ipynb_checkpoints'):
    shutil.rmtree(dataroot + '/.ipynb_checkpoints')

# We can use an image folder dataset the way we have it setup.
dataset = dset.ImageFolder(root=dataroot,
                           transform=transforms.Compose([
                               transforms.CenterCrop(image_size * 12),
                               transforms.Resize(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))


# Create the dataloader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=workers)

fixed_noise = torch.randn(64, nz, 1, 1, device=device)

##################
#                #
# 3 - Optimize   #
#                #
##################



run_tag = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

experiment_id = mlflow.create_experiment(
    f"../../reports/WGAN_Exp_RGB_{run_tag}",
    tags={"version": "v1", "priority": "P1"},
)

mlflow.set_experiment(experiment_id=experiment_id)
study = optuna.create_study(study_name=f"WGAN_study_RGB_{run_tag}", direction="minimize")

func = lambda trial: objective(trial, nz, dataloader, n_epochs = num_epochs, folder = '../../', experiment='WGAN_CadastralRGB_', AlternativeTraining = 50)
study.optimize(func, n_trials=trials)