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
from Generator import OptGen
from Discriminator import OptDis



def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02) # fills the weights? gamma param with normal distribution
        nn.init.constant_(m.bias.data, 0) # fills the bias with the constant 0

def gradient_penalty(D, xr, xf):
    """
    :param D:
    :param xr: [b, 2]
    :param xf: [b, 2]
    :return:
    """
    b_size = tuple(xr.size())
    # [b, 1]
    # t = torch.rand(b_size, 1).cuda()
    t = torch.rand(b_size).cuda()
    # [b, 1] => [b, 2]  broadcasting so t is the same for x1 and x2
    t = t.expand_as(xr)
    # interpolation
    mid = t * xr + (1 - t) * xf
    # set it to require grad info
    mid.requires_grad_()
    
    pred = D(mid)
    grads = autograd.grad(outputs=pred, inputs=mid,
                          grad_outputs=torch.ones_like(pred),
                          create_graph=True, retain_graph=True, only_inputs=True)[0]

    gp = torch.pow(grads.norm(2, dim=1) - 1, 2).mean()

    return gp

def trainModel(netG, netD, device, dataloader, optimizerG, optimizerD, epochs, nz, fixed_noise, folder, AlternativeTraining: int = 0):
    img_list = []
    isDLearning = False
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
        for i, data in enumerate(dataloader, 0):
            xr = data[0].to(device)
            b_size = xr.size(0)
            z = torch.randn(b_size, nz, 1, 1).to(device)
            # 0 Set all gradients for D to 0
            netD.zero_grad()
            # 1.1 train on real data
            predr = netD(xr).view(-1)
            # maximize predr, therefore minus sign
            lossr = -predr.mean() # to be minimized in the optimization, therefore -inf is the goal
            # 1.2 train on fake data
            xf = netG(z).detach()  # without .detach() gradient would be passed down
            predf = netD(xf)
            # minimize predf
            lossf = predf.mean()
            # 1.3 gradient penalty
            gp = 0.2 * gradient_penalty(netD, xr, xf) # lambda gradient penalty = 0.2
            # aggregate all
            loss_D = lossr + lossf + gp 
            if isDLearning or AlternativeTraining == 0 or epoch == 0:
                loss_D.backward()
                optimizerD.step()
            # 2. train G
            netG.zero_grad()
            xf = netG(z)
            predf = netD(xf)
            # maximize predf.mean()
            loss_G = -predf.mean() # to be minimized in the optimization, therefore -1 is the goal
            if not isDLearning or AlternativeTraining == 0:
                loss_G.backward()
                optimizerG.step()
            # Save Losses for plotting later
            # G_losses.append(loss_G.item())
            # D_losses.append(loss_D.item())
        if epoch % 1 == 0:
            print(f'Epoch: {epoch}/{epochs} | D Learn: {isDLearning} | D Loss: {np.round(loss_D.item(), 4)}' + 
                  f'| ErrDReal: {np.round(lossr.item(), 4)} | ErrDFake: {np.round(lossf.item(), 4)} ' + 
                  f'| GradPenality: {np.round(gp.item(), 4)} | G Loss: {np.round(loss_G.item(), 4)}')
        if epoch % 10 == 0:
            # Generate and save fake images to monitor the progress
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()
                i = random.sample(range(b_size), 1)[0]
                if epoch < 10:
                    epoch = '000' + str(epoch)
                elif epoch < 100:
                    epoch = '00' + str(epoch)
                elif epoch < 1000:
                    epoch = '0' + str(epoch)
                path = folder + 'reports/WGANBestImages/WGANProgress_Epoch' + str(epoch) + '.png'
                fake_grid = vutils.make_grid(fake, padding=2, normalize=True)
                save_image(fake_grid, path) # save_image(fake[i], path)
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
    return (img_list)

def test(netG, device, dataloader, nz):
    nz_dim = nz
    errG = []
    for i, data in enumerate(dataloader, 0):
        real = data[0].to(device)
        # batch_size, seq_len = real.size(0), real.size(1)
        batch_size = real.size(0)
        noise = torch.randn(batch_size, nz, 1, 1, device=device)
        mse = nn.MSELoss()
        with torch.no_grad():
            fake = netG(noise)
            ####
            # print(f"dimensions for the fake tensor: {fake.shape}")
            errG += [mse(fake, real)]
    v = torch.tensor(errG).float().mean().item()
    print(f"mse_errG: {v}")
    return v

def suggest_hyperparameters(trial):
    lr = trial.suggest_float("lr", 0.00005, 0.0004) # 1e-4, 1e-3, log=True)
    dropoutG = trial.suggest_float("dropoutG", 0.0, 0.4, step=0.1)
    convsG = trial.suggest_int("convsG", 3, 4, step=1)
    convsD = trial.suggest_int("convsD", 3, 4, step=1)
    return lr, convsG, convsD, dropoutG

def objective(trial: optuna.Trial, nz: int, dataloader, n_epochs: int, folder: 'str', experiment: 'str', AlternativeTraining:int = 0):
    '''
    params
    nz: random unit for the latent space
    dataloader: instance of torchvision.datasets
    n_epochs: number of epochs training the model
    folder: relative path to be added to the destination paths
    experiment: name of the experiment to save the model
    '''
    logging.basicConfig(filename=folder + experiment + '.log')
    best_val_loss = float('Inf')
    nz_dim = nz
    ngpu = torch.cuda.device_count() # Number of GPUs available. Use 0 for CPU mode.
    best_mse_val = None
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
    fixed_noise = torch.randn(64, nz, 1, 1, device=device)
    beta1 = 0.5
    
    with mlflow.start_run():
        
        lr, convsG, convsD, dropoutG = suggest_hyperparameters(trial) # dropoutD, 
        # n_epochs = 50 #1000
        torch.manual_seed(123)
        mlflow.log_params(trial.params)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        mlflow.log_param("device", device)
        
        netD = OptDis(ngpu, convsD).to(device)
        netG = OptGen(ngpu=ngpu, num_conv_layers=convsG, drop_conv2=dropoutG).to(device)
        
        optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999)) # getattr(optim, "Adam")(netD.parameters(), lr=lr)
        optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999)) # getattr(optim, "Adam")(netG.parameters(), lr=lr)
        
        print(f"Convolutions for D: {convsD} | Convolutions for G: {convsG} | LrRate: {np.round(lr,4)} | Dropout G: {dropoutG}")
        
        img_list = trainModel(netG, netD, device, dataloader, optimizerG, optimizerD, n_epochs, nz, fixed_noise, folder, AlternativeTraining = AlternativeTraining)
        mse_errG = test(netG, device, dataloader, nz_dim)
        
        if best_mse_val is None:
            best_mse_val = mse_errG
        if mse_errG <= best_mse_val:
            torch.save(netG, folder + "models/" + experiment + "Generator")
            torch.save(netD, folder + "models/" + experiment + "Discriminator")
            logging.debug(f'Learning Rate: {lr} Convs G: {convsG} | Convs D: {convsD} | Dropout G: {dropoutG}')
            for i, img in enumerate(img_list):
                if i < 10:
                    i = '0' + str(i)
                path = folder + 'reports/WGANBestImages/WGAN' + experiment + 'BestImg_Step' + str(i) + '.png'
                save_image(img, path)
        best_mse_val = min(best_mse_val, mse_errG)
        mlflow.log_metric("mse_errG", mse_errG)

    return best_mse_val
