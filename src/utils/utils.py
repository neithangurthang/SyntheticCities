import torch
import torch.nn as nn
import copy

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
    res = res.view(d).float()
    return res

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02) # fills the weights? gamma param with normal distribution
        nn.init.constant_(m.bias.data, 0) # fills the bias with the constant 0
    return
        

def single_conv(ins: int, kernel_size: int, stride: int = 2, padding: int = 1):
    ''' 
    returns the size of the output after a single convolution given 
    input size, kernel size, stride and padding.
    The operations returns the output size of one dimension (or both dimensions)
    given a quadratic pic and same params for horizontal and vertical processing
    '''
    out = ((ins + 2 * padding - kernel_size) / stride) + 1
    return out

def conv_grid_search(ins: int, kernel_sizes: list[int], strides: list[int], paddings: list[int]):
    '''
    returns all the viable combinations that return entire outputs given a precise input size 
    and list of possible strides, paddings and kernel sizes
    '''
    results = []
    for k in kernel_sizes:
        for p in paddings:
            for s in strides:
                out = single_conv(ins = ins, kernel_size=k, stride = s, padding = p)
                if out.is_integer():
                    result = {'ins': [ins], 'outs': [out], 'kernel_sizes': [k], 'paddings': [p], 'strides': [s]}
                    results.append(result)
    return results

def conv_path_search(ins: int, kernel_sizes: list[int], strides: list[int], paddings: list[int], convs: int = 3):
    '''
    This function returns possible convolution paths to return a vector with dimensions filter, 1, 1
    '''
    solutions = []
    # 
    # results = {'ins': [ins], 'outs': [], 'kernel_sizes': [], 'paddings': [], 'strides': []}
    # candidates = {'ins': [ins], 'outs': [], 'kernel_sizes': [], 'paddings': [], 'strides': []}
    candidates = conv_grid_search(ins, kernel_sizes, strides, paddings)
    for conv in range(convs - 1):
        new_candidate_list = []
        for c in candidates:
            # print(f"candidate: {c}")
            new_ins = c['outs'][-1]
            new_candidates = conv_grid_search(new_ins, kernel_sizes, strides, paddings)
            # print(f"new candidates: {new_candidates}")
            for new in new_candidates:
                # thread = dict(c)
                thread = copy.deepcopy(c)
                # print(f"thread: {thread}")
                thread['ins'].append(new['ins'][0])
                thread['outs'].append(new['outs'][0])
                thread['kernel_sizes'].append(new['kernel_sizes'][0])
                thread['paddings'].append(new['paddings'][0])
                thread['strides'].append(new['strides'][0])
                new_candidate_list.append(thread)
                # print(f'c: {c}')
            candidates.remove(c)
        # print(f"new candidates list: {new_candidate_list}")
        candidates = new_candidate_list[:]
        # print(f"candidates: {candidates}")
        # print(f"candidates: {candidates}")
    for cand in candidates:
        if cand['outs'][-1] == 1:
            solutions.append(cand)
    return solutions, candidates