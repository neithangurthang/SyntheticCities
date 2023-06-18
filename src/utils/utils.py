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
        
###
### FIND THE PARAMS FOR THE CONVOLUTIONAL LAYERS
###

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

def conv_path_search(ins: int, kernel_sizes: list[int], strides: list[int], paddings: list[int], convs: int = 3, out: int = 1, verbose:bool = False):
    '''
    This function returns possible convolution paths to return a vector with dimensions filter, 1, 1
    '''
    solutions = []
    candidates = conv_grid_search(ins, kernel_sizes, strides, paddings)
    for conv in range(convs - 1):
        new_candidate_list = []
        for c in candidates:
            new_ins = c['outs'][-1]
            new_candidates = conv_grid_search(new_ins, kernel_sizes, strides, paddings)
            for new in new_candidates:
                thread = copy.deepcopy(c)
                thread['ins'].append(new['ins'][0])
                thread['outs'].append(new['outs'][0])
                thread['kernel_sizes'].append(new['kernel_sizes'][0])
                thread['paddings'].append(new['paddings'][0])
                thread['strides'].append(new['strides'][0])
                new_candidate_list.append(thread)
                # print(f'c: {c}')
            candidates.remove(c)
        candidates = new_candidate_list[:]
    for cand in candidates:
        res = test_candidate(candidate = cand, out = out, verbose = verbose)
        if res:
            solutions.append(res)
    return solutions, candidates

def test_candidate(candidate: dict, out: int, verbose = False):
    '''
    This function tests if a candidate complies with the criteria
    
    '''
    # condition 1: returns dimension 1
    condition1 = candidate['outs'][-1] == out
    # condition 2: kernel size larger or equal to stride
    isStrideSmallerKernel = True
    # condition 3: decreasing dimension of the kernel
    isKernelDecreasging = True
    # condition 4: decreasing stride
    isStrideDecreasing = True
    # condition 5: decreasing paddings
    isPaddingDecreasing = True
    for i in range(len(candidate['strides'])):
        if candidate['strides'][i] > candidate['kernel_sizes'][i]:
            isStrideSmallerKernel = False
        if i > 0:
            if candidate['kernel_sizes'][i] > candidate['kernel_sizes'][i - 1]:
                isKernelDecreasging = False
            if candidate['strides'][i] > candidate['strides'][i - 1]:
                isStrideDecreasing = False
            if candidate['paddings'][i] > candidate['paddings'][i - 1]:
                isPaddingDecreasing = False
    conditions = condition1 and isStrideSmallerKernel and isKernelDecreasging and isStrideDecreasing and isPaddingDecreasing
    if conditions:
        return candidate
    else:
        if verbose:
            print('#'*20)
            print(candidate)
            if not condition1:
                print(f'Output {candidate["outs"][-1]} different than target: {out}')
            if not isStrideSmallerKernel:
                print(f'Strides {candidate["strides"]} larger than kernel sizes: {candidate["kernel_sizes"]}')
            if not isKernelDecreasging:
                print(f'Kernel size is not decreasing: {candidate["kernel_sizes"]}')
            if not isStrideDecreasing:
                print(f'Stride is not decreasing: {candidate["strides"]}')
            if not isPaddingDecreasing:
                print(f'Padding is not decreasing: {candidate["paddings"]}')
            return None

######
###### REPRESENT PROBABILITIES AS RGB COLORS
######

def pixel_to_class(values: torch.Tensor):
    '''
    returns the rgb class values of a certain pixel with the highest probability
    img: tensor with channels, pixel raws, pixel columns
    x, y: position of the selected pixel
    channel 1: buildings -> rgb (0, 0, 0) -> black
    channel 2: roads -> rgb (1, 1, 1) -> white
    channel 3: other -> rgb(0.4118, 0.4118, 0.4118) -> dimgray
    '''
    l = values.max()
    ind = [i for i, j in enumerate(values) if j == l][0]
    result = torch.Tensor()
    if ind == 0:
        result = torch.Tensor([0,0,0]) # building  
    elif ind == 1:
        result = torch.Tensor([1,1,1]) # road
    elif ind == 2:
        result = torch.Tensor([0.4118,0.4118,0.4118]) # other
    return result

def img_to_class(img: torch.Tensor):
    '''
    postprocesses an image generated by a GNet
    '''
    view = img.view(-1, 3)
    res = torch.stack([pixel_to_class(x_i) for x_i in torch.unbind(view, dim=0)], dim=0)
    res = res.reshape(64,64,3)
    res = res.permute(2,0,1)
    return res

def batch_to_class(batch: torch.Tensor):
        '''
        postprocess a batch of images generated by a GNet interpreting the probability to have a certain class
        '''
        res = torch.stack([
        img_to_class(x_i) for i, x_i in enumerate(torch.unbind(batch, dim=0), 0)], dim=0) 
        return res

def postprocess_rgb(img: torch.tensor):
    '''
    post processes a rgb tensor and returns a tensor of the same shape
    where only one channel has a 1 and the others a 0 -> the winner takes all
    '''
    img = img.permute(1, 2, 0) # x, y, channel
    for x in range(img.size(0)):
        for y in range(img.size(1)):
            pixel = img[x][y].tolist()
            i = pixel.index(max(pixel))
            img[x][y][0:3] = 0
            img[x][y][i] = 1
    return(img)
            