import sys
import os
import random
import numpy as np
import datetime
import logging

import torch
import torch.nn as nn
from torch import autograd
from tqdm.auto import tqdm
from torchvision import transforms
# from torchvision.datasets import MNIST
from torchvision.utils import make_grid, save_image
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
from OptimisationFunctions import gradient_penalty

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os


def getGreyscaleLabels(img: torch.tensor, threshold: int = 0.1):
    white = ((img < threshold).sum().item()/img.numel())
    black = ((img > 1 - threshold).sum().item()/img.numel())
    grey = (1 - white - black)
    labels = torch.tensor([white, grey, black]).view(-1, 3)
    return labels


def trainCGANModel(netG, netD, device: torch.device, dataloader: torch.utils.data.dataloader.DataLoader, 
               optimizerG, optimizerD, 
               fixed_noise:torch.Tensor, fixed_labels: torch.tensor, folder: str, epochs: int = 10, nz: int = 100,
               experiment: str = 'WGANRGB',
               AlternativeTraining: int = 0, logger: logging.Logger = None):
    '''
    Params:
        netG: Generator.OptGen
        netD: Discriminator
        device: cuda, cpu, ...
        dataloader: dataloader to feed the learning process
        optimizerG: optimizer for the netG
        optimizerD: optimizer for the netD
        epochs: number of epochs to train the model
        nz: dimension of the noise tensor
        fixed_noise: tensor with fixed noise to evaluate the progress of the generator
        folder: path where to save the produced pictures and the log
        experiment: prefix for the pictures
        AlternativeTraining: number of epochs in which netG and netD are trained exclusively. 
            If 0, then both netD and netG are trained at the same time
        logger: logger to save the logging
        
        Returns the pictures generated by netG after n steps and optimizes both the parameters
        of netG and netD by training them using the Wasserstein distance and the gradient penalty
    
    '''
    TRAIN_G_EVERY = 2  # training generator only every 3d epoch comparing to discriminator
    # Creating a pool of noises
    # z_pool = [torch.randn(1, nz, 1, 1).to(device) for _ in range(1000)]  # FIXME: need to make it dependent on size of real data
    # logging.basicConfig(filename = folder + 'trainingGANs.log', level = logging.DEBUG) 
    img_list = []
    isDLearning = False
    # if logger:
        # logger.debug(f'Training GANS with NetG[conv. layers: {netG.num_conv_layers}, drop out: {netG.drop_conv2}] and NetD[conv. layers: {netD.num_conv_layers}]')
    for epoch in range(epochs):
        # switch between training G and D
        if isDLearning and AlternativeTraining > 0:
            if ((epoch + 1) % AlternativeTraining) == 0:
                isDLearning = False
        elif AlternativeTraining > 0:
            if ((epoch + 1) % AlternativeTraining) == 0:
                isDLearning = True
        else:
            isDLearning = True
        # print(AlternativeTraining, isDLearning)
        netG.train()
        netD.train()
        for i, data in enumerate(dataloader, 0):
            xr = data[0].to(device)
            # print(f'size of xr: {xr.size()}')
            b_size = xr.size(0)
            z = torch.randn(b_size, nz, 1, 1).to(device)
            l = torch.rand([b_size, 3], device = device)
            # z = torch.cat(random.sample(z_pool, b_size))
            # 0 Set all gradients for D to 0
            netD.zero_grad()
            # 1.1 train on real data
            predr = netD(xr).view(-1)
            # maximize predr, therefore minus sign
            lossr = -predr.mean() # to be minimized in the optimization, therefore -inf is the goal
            # 1.2 train on fake data
            xf = netG(z, l).detach() # .detach() not needed here
            # Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
            # RuntimeError: Trying to backward through the graph a second time (or directly access saved tensors after they have already been freed). Saved 
            # intermediate values of the graph are freed when you call .backward() or autograd.grad(). Specify retain_graph=True if you need to backward through 
            # the graph a second time or if you need to access saved tensors after calling backward.
            predf = netD(xf)
            # minimize predf
            lossf = predf.mean()
            # 1.3 gradient penalty
            # gp = torch.tensor([0])
            # gp = 1e+1 * gradient_penalty(netD, xr, xf) # lambda gradient penalty = 0.2
            # aggregate all
            loss_D = lossr + lossf # + gp 
            if isDLearning or AlternativeTraining == 0 or epoch == 0:
                loss_D.backward()
                optimizerD.step()
                for p in netD.parameters():
                    p.data.clamp_(-0.01, 0.01)
            # 2. train G
            netG.zero_grad()
            xf = netG(z, l)
            predf = netD(xf) # predf with new params, since netD has already been trained
            # maximize predf.mean()
            loss_G = -predf.mean() # to be minimized in the optimization, therefore -1 is the goal
            if not isDLearning or AlternativeTraining == 0:
            # if i % TRAIN_G_EVERY == 0:
                loss_G.backward()
                optimizerG.step()
        if epoch % 100 == 0:
            netG.eval()
            netD.eval()
            print(f'Epoch: {epoch}/{epochs} | D Learn: {isDLearning} | D Loss: {np.round(loss_D.item(), 4)} ' + 
                  f'| ErrDReal: {np.round(lossr.item(), 4)} | ErrDFake: {np.round(lossf.item(), 4)} ' + 
                  f'| G Loss: {np.round(loss_G.item(), 4)}') # | GradPenality: {np.round(gp.item(), 4)} 
            # SAVING MODEL
            torch.save(netG, folder + "models/" + experiment + "_NetG_Training")
            torch.save(netD, folder + "models/" + experiment + "_NetD_Training")
            with torch.no_grad():
                fake = netG(fixed_noise, fixed_labels).detach().cpu()
                i = random.sample(range(b_size), 1)[0]
                if epoch < 10:
                    epoch = '000' + str(epoch)
                elif epoch < 100:
                    epoch = '00' + str(epoch)
                elif epoch < 1000:
                    epoch = '0' + str(epoch)
                path = folder + 'reports/CGANBestImages/' + experiment +'_Epoch_' + str(epoch) + '.png'
                fake_grid = vutils.make_grid(fake, padding=2, normalize=True)
                save_image(fake_grid, path) # save_image(fake[i], path)
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
                if logger:
                    logger.debug(f'Epoch: {epoch} | Error G: {loss_G} | Error D: {loss_D}')
    return (img_list)