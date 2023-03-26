# 1. Import Libraries
#####################################################################
#                                                                   #
#      This code optimises G and D on custom cadastral images       #
#                                                                   #
#####################################################################


from __future__ import print_function
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel # implements data parallelism across multiple devices/GPUs
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim # optimizer for example optim.Adam([var1, var2], lr=0.0001)
import torch.utils.data # for the dataloader class (python iterable over a dataset)
import torchvision.datasets as dset # all datasets available in torch vision. not sure if needed here
import torchvision.transforms as transforms # all transformations for data augmentation (pad, resize, centercrop, ...)
import torchvision.utils as vutils # draw bounding box, segmantation mask, keypoints. convert to rgb, make grid, save_image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import optuna
from optuna.trial import TrialState
import logging
from IPython.display import HTML # to embed html in the Ipython output
import datetime
import mlflow

logging.basicConfig(level=logging.DEBUG)
logging.basicConfig(filename='../../reports/MVPOptimizer.log') 

# Set random seed for reproducibility
manualSeed = 999
#manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

# 2. Set the params

dataroot = "../../data/cadastralExport" # Root directory for train dataset
datarootTest = "../../data/cadastralExport" # Root directory for dataset
workers = 2 # Number of workers for dataloader
batch_size = 64 # Batch size during training
image_size = 64 # Spatial size of training images. All images will be resized to this
nc = 3 # Number of channels in the training images. For color images this is 3
nz = 100 # Size of z latent vector (i.e. size of generator input)
ngf = 64 # Size of feature maps in generator
ndf = 64 # Size of feature maps in discriminator
num_epochs = 50 # Number of training epochs
lr = 0.0002 # Learning rate for optimizers
beta1 = 0.5 # Beta1 hyperparam for Adam optimizers
ngpu = torch.cuda.device_count() # Number of GPUs available. Use 0 for CPU mode.
# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

# 3. Create Dataloader

# We can use an image folder dataset the way we have it setup.
dataset = dset.ImageFolder(root=dataroot,
                           transform=transforms.Compose([
                               transforms.RandomCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
# Create the dataloaders
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=workers)

datasetTest = dset.ImageFolder(root=datarootTest,
                               transform=transforms.Compose([
                               transforms.RandomCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
# Create the dataloader for test: at the moment the same folder as train
dataloaderTest = torch.utils.data.DataLoader(datasetTest, batch_size=batch_size,
                                             shuffle=True, num_workers=workers)

# 4. Weight Initialization
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02) # fills the weights? gamma param with normal distribution
        nn.init.constant_(m.bias.data, 0) # fills the bias with the constant 0
        
# 5. Generator Class
class OptGen(nn.Module):
    def __init__(self, ngpu, num_conv_layers):
        super(OptGen, self).__init__()
        self.ngpu = ngpu
        self.num_filters = [3] 
        self.num_filters.extend([2**(i+6) for i in range(num_conv_layers-1)])
        self.strides = [2]
        self.paddings = [1]
        self.kernelSizes = [4]
        self.out_size = []
        if num_conv_layers == 3:
            self.strides.extend([2,2])
            self.paddings.extend([0,0])
            self.kernelSizes.extend([14,10])
        if num_conv_layers == 4:
            self.strides.extend([2,2,2])
            self.paddings.extend([1,0,0])
            self.kernelSizes.extend([4,6,6])
        if num_conv_layers == 5:
            self.strides.extend([2,2,2,2])
            self.paddings.extend([1,1,1,0])
            self.kernelSizes.extend([4,4,4,4])
        if num_conv_layers == 6:
            self.strides.extend([2,2,2,2,2])
            self.paddings.extend([1,1,1,1,1])
            self.kernelSizes.extend([4,4,4,4,4])
        # self.out_size = []
        # same scheme as for DNet, but inverted
        self.num_filters.reverse()
        self.strides.reverse()
        self.paddings.reverse()
        self.kernelSizes.reverse()
        
        self.main = nn.Sequential(
            # input is Z, going into a convolution with dimensions c=nz, h=1, w=1
            # output size: (2**6) x 4 x 4 O=(I-1)*s+k-2p
            nn.ConvTranspose2d(in_channels=nz, #deconvolution!
                               out_channels=self.num_filters[0], #ngf * 8, 
                               kernel_size=self.kernelSizes[0], 
                               stride=self.strides[0], 
                               padding=self.paddings[0], 
                               bias=False), # (1-1)*1+4-2*0=4
            nn.BatchNorm2d(self.num_filters[0]),
            nn.ReLU(True)
        )
        self.out_size.append([self.num_filters[0], (1-1)*self.strides[0]+self.kernelSizes[0]-2*self.paddings[0]])
        self.num_modules = 3
        for i in range(1, num_conv_layers):
            self.main.add_module(str(3*i-1)+"): TransConv_"+str(i+1), nn.ConvTranspose2d(in_channels=self.num_filters[i-1],
                                                            out_channels=self.num_filters[i],
                                                            kernel_size=self.kernelSizes[i],
                                                            stride=self.strides[i],
                                                            padding=self.paddings[i],
                                                            bias=False))
            self.out_size.append([self.num_filters[i], (self.out_size[i-1][1]-1)*self.strides[i]+self.kernelSizes[i]-2*self.paddings[i]])
            self.num_modules += 1
            if i + 1 < num_conv_layers: 
                self.main.add_module(str(3*i)+"): BatchNorm_" + str(i+1), nn.BatchNorm2d(self.num_filters[i]))
                self.main.add_module(str(1+3*i)+"): ReLU_" + str(i+1), nn.ReLU(True))
                self.num_modules += 2
            
        self.main.add_module(str(self.num_modules), nn.Tanh())
        ## print(f"Progression of the sizes in the deconvolution: {self.out_size}")
    
    def forward(self, input):
        return self.main(input)

# 6. Discriminator Model

class OptDis(nn.Module):
    def __init__(self, ngpu, num_conv_layers):
        super(OptDis, self).__init__()
        self.ngpu = ngpu
        self.num_filters = [2**(i+6) for i in range(num_conv_layers-1)]
        self.num_filters.append(1)
        self.strides = [2]
        self.paddings = [1]
        self.kernelSizes = [4]
        self.out_size = []
        if num_conv_layers == 3:
            self.strides.extend([2,2])
            self.paddings.extend([0,0])
            self.kernelSizes.extend([14,10])
        if num_conv_layers == 4:
            self.strides.extend([2,2,2])
            self.paddings.extend([1,0,0])
            self.kernelSizes.extend([4,6,6])
        if num_conv_layers == 5:
            self.strides.extend([2,2,2,2])
            self.paddings.extend([1,1,1,0])
            self.kernelSizes.extend([4,4,4,4])
        if num_conv_layers == 6:
            self.strides.extend([2,2,2,2,2])
            self.paddings.extend([1,1,1,1,1])
            self.kernelSizes.extend([4,4,4,4,4])
            
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64 -> output nc = 2**6 x 32 x 32
            nn.Conv2d(in_channels=nc, 
                      out_channels=self.num_filters[0], 
                      kernel_size=self.kernelSizes[0], 
                      stride=self.strides[0], 
                      padding=self.paddings[0], 
                      bias=False), # (64-4+2)/2 + 1=32
            nn.LeakyReLU(0.2, inplace=True))
        for i in range(1, num_conv_layers):
            # input is nc=(2**(i+5) x 32 x 32 -> output nc = 2**(i+6) x 32*2**(-i) x 32*2**(-i)
            self.main.add_module(str(3*i-1)+"): Conv_"+str(i+1), nn.Conv2d(in_channels=self.num_filters[i-1],
                                                            out_channels=self.num_filters[i],
                                                            kernel_size=self.kernelSizes[i],
                                                            stride=self.strides[i],
                                                            padding=self.paddings[i],
                                                            bias=False))
            if i + 1 < num_conv_layers: 
                self.main.add_module(str(3*i)+"): BatchNorm_" + str(i+1), nn.BatchNorm2d(self.num_filters[i]))
                self.main.add_module(str(1+3*i)+"): LeakyReLU_" + str(i+1), nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.main.add_module(str(3*i)+"): Sigmoid", nn.Sigmoid())
    
    def forward(self, input):
        return self.main(input)

# 7. Hyperparameter Tuning
# Initialize BCELoss function
criterion = nn.BCELoss() #binary cross entropy, adapt for a probability output and for sigmoid function in D

# Create batch of latent vectors that we will use to visualize
#  the progression of the generator
fixed_noise = torch.randn(64, nz, 1, 1, device=device)

# Establish convention for real and fake labels during training
real_label = 1.
fake_label = 0.

def train(netG, netD, device, dataloader, optimizerG, optimizerD, epochs):
    
    img_list = []
    G_losses = []
    D_losses = []
    iters = 0

    print("Starting Training Loop...")
    # For each epoch
    for epoch in range(epochs):
        # For each batch in the dataloader
        for i, data in enumerate(dataloader, 0):

            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            ## Train with all-real batch
            netD.zero_grad() # sets all gradients to 0
            # Format batch
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0) # number of images, usually=128, last batch=23
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device) # size, fill value
            # Forward pass real batch through D
            output = netD(real_cpu).view(-1)
            # Calculate loss on all-real batch
            errD_real = criterion(output, label)
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
            errD_fake = criterion(output, label)
            # Calculate the gradients for this batch, accumulated (summed) with previous gradients
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            # Compute error of D as sum over the fake and the real batches
            errD = errD_real + errD_fake
            # Update D
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = netD(fake).view(-1)
            # Calculate G's loss based on this output
            errG = criterion(output, label)
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            # Update G
            optimizerG.step()

            # Output training stats
            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (epoch, num_epochs, i, len(dataloader),
                         errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            # Check how the generator is doing by saving G's output on fixed_noise
            if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

            iters += 1

def test(netG, device, dataloader):
    nz_dim = nz
    errG = []
    for i, data in enumerate(dataloader, 0):
        real = data[0].to(device)
        batch_size = real.size(0)
        noise = torch.randn(batch_size, nz, 1, 1, device=device)
        mse = nn.MSELoss()
        with torch.no_grad():
            fake = netG(noise)
            errG += [mse(fake, real)]
    v = torch.tensor(errG).float().mean().item()
    return v

def suggest_hyperparameters(trial):
    lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
    optimizer_name = trial.suggest_categorical("optimizer_name", ["Adam", "Adadelta"]) # 
    convsG = trial.suggest_int("convsG", 3, 6, step=1)
    convsD = trial.suggest_int("convsD", 3, 6, step=1)
    return lr, optimizer_name, convsG, convsD

def objective(trial: optuna.Trial):
    best_val_loss = float('Inf')
    nz_dim = nz
    best_mse_val = None

    with mlflow.start_run():

        lr, optimizer_name, convsG, convsD = suggest_hyperparameters(trial)
        n_epochs = 5000
        torch.manual_seed(123)
        mlflow.log_params(trial.params)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        mlflow.log_param("device", device)
        
        netD = OptDis(ngpu, convsD).to(device)
        netG = OptGen(ngpu, convsG).to(device)
        
        netD.apply(weights_init)
        netG.apply(weights_init)
        
        optimizerD = getattr(optim, optimizer_name)(netD.parameters(), lr=lr) # Add lambda 
        optimizerG = getattr(optim, optimizer_name)(netG.parameters(), lr=lr) # Add lambda

        train(netG, netD, device, dataloader, optimizerG, optimizerD, n_epochs)
        mse_errG = test(netG, device, dataloaderTest)
        
        if best_mse_val is None:
            best_mse_val = mse_errG
        best_mse_val = min(best_mse_val, mse_errG)
        mlflow.log_metric("mse_errG", mse_errG)

    return best_mse_val

run_tag = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M")

experiment_id = mlflow.create_experiment(
    f"../../reports/TSGAN_Exp_{run_tag}",
    tags={"version": "v1", "priority": "P1"},
)

mlflow.set_experiment(experiment_id=experiment_id)
study = optuna.create_study(study_name=f"TSGAN_study_{run_tag}", direction="minimize")
study.optimize(objective, n_trials=200)

print(study.best_trial.value)  # Show the best value.
