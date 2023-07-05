"""
this code optimises G and D on RGB cadastral images with 64 x 64 px resolution

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
from torchvision.utils import make_grid
from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as dset # all datasets available in torch vision. not sure if needed here
import torchvision.utils as vutils # draw bounding box, segmantation mask, keypoints. convert to rgb, make grid, save_image
import torch.optim as optim # optimizer for example optim.Adam([var1, var2], lr=0.0001)

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os

import optuna
from optuna.trial import TrialState
import mlflow

import matplotlib.pyplot as plt
torch.manual_seed(0) # Set for testing purposes, please do not change!
import matplotlib.animation as animation
from IPython.display import HTML # to embed html in the Ipython output

sys.path.append("../../src/models/")
sys.path.append("../../src/utils/")

from CGANGNet64_GS import CondGen, ConditionNetG
from CGANDNet64 import CondDis, Condition
from utils import normalizeRGB
from utils import weights_init
from OptimisationCGAN import trainCGANModel
from OptimisationFunctions import suggest_hyperparameters, trainModel, test, objective
from ddp_utils import ddp_setup, prepare_dataloader

#######################
#                     #
# 1 - Define Params   #
#                     #
#######################

# dataroot = "../../../cadastralExportRGBSelected" # Root directory for train dataset
dataroot = "../../../cadastralExport/"
workers = 4 # Number of workers for dataloader
batch_size = 20 # Batch size during training
image_size = 64 # Spatial size of training images. All images will be resized to this
nc = 3 # Number of channels in the training images. For color images this is 3
noise_filters = 1 # noise filters
noise_pixels = 8 # noise pixels
nz = noise_filters*noise_pixels*noise_pixels # noise for one single image
ngf = 64 # Size of feature maps in generator
ndf = 64 # Size of feature maps in discriminator
num_epochs = 1500 # Number of training epochs
trials = 20 # number of trials
AltTrain = 0 # epochs alternative training 
lr = 0.0002 # Learning rate for optimizers
beta1 = 0.9 # Beta1 hyperparam for Adam optimizers
lambda_gradient_penality = 0.2 # to adjust the Wasserstein distance with interpolation between real and fake data
ngpu = torch.cuda.device_count() # Number of GPUs available. Use 0 for CPU mode.
# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
experiment = 'CGAN_Greyscale64'
new_experiment = False

# logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s:%(levelname)s:%(message)s")
file_handler = logging.FileHandler('../../reports/CGAN_CadastralGS64.log')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

#####################
#                   #
# 3 - LOAD DATASET  #
#                   #
#####################
        
# first delete the unnecessary folder 
if os.path.exists(dataroot + '/.ipynb_checkpoints'):
    shutil.rmtree(dataroot + '/.ipynb_checkpoints')

# We can use an image folder dataset the way we have it setup.
dataset = dset.ImageFolder(root=dataroot,
                           transform=transforms.Compose([
                               transforms.RandomRotation(degrees=(0, 180), expand = False),
                               transforms.CenterCrop(image_size * 10),
                               transforms.Resize(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               transforms.Grayscale(num_output_channels=1)
                           ]))

# Create the dataloader

dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=workers)

fixed_noise = torch.randn(batch_size, nz, 1, 1, device=device)
fixed_labels = torch.rand([batch_size, 3], device = device)

##################
#                #
# 3 - TRAIN      #
#                #
##################

opt_params = {'lr': 1e-4, 'dropoutG': 0.3, 'convsG': 3, 'convsD': 3}

best_lr = opt_params['lr']
best_dropooutG = opt_params['dropoutG']

torch.manual_seed(23)
np.random.seed(23)

path_trnD = '../../models/' + experiment + '_NetD_Training'
path_endD = '../../models/' + experiment + '_NetD_Trained'
path_trnG = '../../models/' + experiment + '_NetG_Training'
path_endG = '../../models/' + experiment + '_NetG_Trained'

if new_experiment or not (os.path.isfile(path_endD) or os.path.isfile(path_trnD)):
    print(f'Create a new DNet, weights initialized')
    netD = CondDis(num_conv_layers=opt_params['convsD'])
    netD.apply(weights_init)
else:
    if os.path.isfile(path_endD):
        print(f'loading trained DNet from {path_endD}')
        netD = torch.load(path_endD)
    elif os.path.isfile(path_trnD):
        print(f'loading trained DNet from {path_trnD}')
        netD = torch.load(path_trnD)
if new_experiment or not (os.path.isfile(path_endG) or os.path.isfile(path_trnG)):
    print(f'Create a new GNet, weights initialized')
    netG = CondGen(nz=nz, num_conv_layers=opt_params['convsG'], drop_conv2=opt_params['dropoutG'], 
                 labels_numb = 3) #
    netG.apply(weights_init)
else:
    if os.path.isfile(path_endG):
        print(f'loading trained GNet from {path_endG}')
        netG = torch.load(path_endG)
    elif os.path.isfile(path_trnG):
        print(f'loading trained GNet from {path_trnG}')
        netG = torch.load(path_trnG)

# If more devices are available
if torch.cuda.device_count() > 1:
    netG = nn.DataParallel(netG)
    netD = nn.DataParallel(netD)

netG.to(device)
netD.to(device)

optimizerG = optim.Adam(netG.parameters(), lr=best_lr) # betas = 0.9, 0.99 by default -> ok
optimizerD = optim.Adam(netD.parameters(), lr=best_lr) # betas = 0.9, 0.99 by default -> ok

img_list_greyscale = trainCGANModel(netG=netG, netD=netD, 
                                device = device, dataloader = dataloader, optimizerG = optimizerG,
                                optimizerD = optimizerD, fixed_noise=fixed_noise, fixed_labels = fixed_labels, folder='../../',  
                                epochs=num_epochs, nz=nz, experiment = experiment, AlternativeTraining=AltTrain,
                                logger=logger)

torch.save(netG, '../../models/' + experiment + "_NetG_Trained")
torch.save(netD, '../../models/' + experiment + "_NetD_Trained")

##################
#                #
# 4 - Optimize   #
#                #
##################
## 
## run_tag = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
## 
## experiment_id = mlflow.create_experiment(
##     f"../../reports/WGAN_RGB64_{run_tag}",
##     tags={"version": "v1", "priority": "P1"},
## )
## 
## mlflow.set_experiment(experiment_id=experiment_id)
## study = optuna.create_study(study_name=f"WGAN_RGB64_{run_tag}", direction="minimize")
## 
## func = lambda trial: objective(trial, nz = nz, dataloader=dataloader, 
##                                n_epochs = num_epochs, folder = '../../', 
##                                experiment='WGANRGB', AlternativeTraining = 0, 
##                                logger = logger)
## study.optimize(func, n_trials=trials)
## 
## best_trial = study.best_trial
## 
## print(f"Trial with highest accuracy: ")
## print(f"\tnumber: {best_trial.number}")
## print(f"\tparams: {best_trial.params}")
## print(f"\tvalues: {best_trial.values}")
## 
## logger.debug(f"Trial with highest accuracy: ")
## logger.debug(f"\tnumber: {best_trial.number}")
## logger.debug(f"\tparams: {best_trial.params}")
## logger.debug(f"\tvalues: {best_trial.values}")