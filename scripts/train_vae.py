# Copyright 2019 Stanislav Pidhorskyi
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#  http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import print_function
import shutil
from scipy import misc
from torch import optim
from torchvision.utils import save_image
import numpy as np
import pickle
import time
import random
import os
import sys
sys.path.append('../')

from src.models.VAE import *
from dlutils import batch_provider
from dlutils.pytorch.cuda_helper import *

im_size = 300
batch_size = 128
DEVICE = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")  # which device to run on

# We can use an image folder dataset the way we have it setup
if os.path.exists(DATAROOT + '/.ipynb_checkpoints'):
    shutil.rmtree(DATAROOT + '/.ipynb_checkpoints')
    
# dataset = load_folder(DATAROOT, resolution=(IMG_SIZE, IMG_SIZE), mult=MULT, device='cpu')

# dataset = torch.load('../data/dataset.pkl')
dataset = torch.load('../data/roads.pkl')

dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
                                             shuffle=True, num_workers=WORKERS, drop_last=True)


def loss_function(recon_x, x, mu, logvar):
    BCE = torch.mean((recon_x - x)**2)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.mean(torch.mean(1 + logvar - mu.pow(2) - logvar.exp(), 1))
    return BCE, KLD * 0.1


def main():
    z_size = 100
    vae = VAE(zsize=z_size, layer_count=5)
    vae.to(DEVICE)
    vae.train()
    vae.weight_init(mean=0, std=0.02)

    lr = 0.0005

    vae_optimizer = optim.Adam(vae.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=1e-5)
 
    num_epochs = 40

    fixed_noise = torch.randn(64, z_size, 1, 1, device=DEVICE) 

    for epoch in range(num_epochs):
        vae.train()

        rec_loss = 0
        kl_loss = 0

        epoch_start_time = time.time()

        if (epoch + 1) % 8 == 0:
            vae_optimizer.param_groups[0]['lr'] /= 4
            print("learning rate change!")

        i = 0
        for i, x in enumerate(dataloader):
            vae.train()
            vae.zero_grad()
            rec, mu, logvar = vae(x)

            loss_re, loss_kl = loss_function(rec, x, mu, logvar)
            (loss_re + loss_kl).backward()
            vae_optimizer.step()
            rec_loss += loss_re.item()
            kl_loss += loss_kl.item()

            #############################################

            # os.makedirs('results_rec', exist_ok=True)
            # os.makedirs('results_gen', exist_ok=True)

            epoch_end_time = time.time()
            per_epoch_ptime = epoch_end_time - epoch_start_time

            # report losses and save samples each 60 iterations
            m = 60
            if i % m == 0:
                rec_loss /= m
                kl_loss /= m
                print('\n[%d/%d] - ptime: %.2f, rec loss: %.9f, KL loss: %.9f' % (
                    (epoch + 1), num_epochs, per_epoch_ptime, rec_loss, kl_loss))
                rec_loss = 0
                kl_loss = 0
                with torch.no_grad():
                    vae.eval()
                    fake = vae.decode(fixed_noise).cpu()
                    fake_grid = vutils.make_grid(fake, padding=2, normalize=True)
                    vutils.save_image(fake_grid, f'../reports/vae/sample_{epoch}_{i}.png')

    print("Training finish!... save training results")
    torch.save(vae, "../models/vae.pkl")

if __name__ == '__main__':
    main()