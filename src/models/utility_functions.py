import torch
import torch.nn as nn

def normalizeRGB(t: torch.tensor):
    """ 
    This function takes a tensor with one or more images
    and returns a new one with values scaled between 0 and 1 
    so that these can be plot with matplotlib
    """
    d = list(t.size())
    res = t.view(t.size(0), -1)
    res -= res.min(1, keepdim=True)[0]
    res /= res.max(1, keepdim=True)[0]
    res = res.view(d)
    return res

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02) # fills the weights? gamma param with normal distribution
        nn.init.constant_(m.bias.data, 0) # fills the bias with the constant 0