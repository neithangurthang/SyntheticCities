from os import listdir
from os.path import isfile, join
import torch
import torch.nn.functional as F
from typing import Optional
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt


class WrongResolutionException(Exception):
    "Raised when the resoltuion is bigger than the initial image size"
    pass


class CadastralImage:
    """
    Holding the image of cadastral map along with it's vector representation
    """
    def __init__(self, src: str, resolution: Optional[tuple] = None, mult: Optional[int] = 11, device: Optional[str] = 'cpu'):
        self.src = src
        self._load_transform(resolution, mult, device)
        
    def _load_transform(self, resolution: Optional[tuple] = None, mult: Optional[int] = 11, device: Optional[str] = 'cpu'):
        """
        Loads and transforms the image into 3-D channel representation (with resize if necessary)
        """
        self.image = Image.open(self.src).convert('RGB')  # use 'L' convertion when grayscale is required
        self.vec = transforms.functional.to_tensor(self.image).to(device)
        # case image is too big we'll resize it according to the resolution provided
        if resolution and resolution != self.image.size:
            if resolution[0] > self.image.size[0] or resolution[1] > self.image.size[1]:
                raise WrongResolutionException(f"Cannot turn image of size {self.image.size} into {resolution}")
            else:
                croper = transforms.CenterCrop(resolution[0] * mult)
                self.vec = croper.forward(self.vec)
                resizer = transforms.Resize(resolution[0])
                self.vec = resizer.forward(self.vec)
        
        self.vec = F.one_hot(torch.argmax(self.vec, dim=0, keepdim=True)[0]).permute(2, 0, 1).float()
            
    def _grayscale_to_rgb(self):
        # leaving only first dimension after converting if grayscale
        self.vec = self.vec.reshape(self.vec.shape[1], self.vec.shape[2])
        # leaving only 0.5 value in case of non-binary element
        # self.vec = torch.where((self.vec == 0) + (self.vec == 1), self.vec, 0.5)
        
        # Converting an image to 3-channel RGB (roads-greens-buildings)
        r = torch.zeros_like(self.vec)
        r[self.vec == True] = 1

        b = torch.zeros_like(self.vec)
        b[self.vec == False] = 1

        g = torch.ones_like(self.vec)
        g = g - r - b
        
        self.vec = torch.tensor([r.numpy(), g.numpy(), b.numpy()])
        
    def show_image(self):
        """
        Showing the image.
        
        NB! If you want to see the original image (not resized), call `self.image` itself in a command line
        """
        plt.imshow(self.vec.cpu().permute(1, 2, 0))  # required for correct display of RGB images
        

def load_folder(src: str, resolution: Optional[tuple] = None, mult: Optional[int] = 11, device: Optional[str] = 'cpu'):
    """
    Loading all the files from given folder
    """
    dataset = []
    onlyfiles = [f for f in listdir(src) if isfile(join(src, f))]
    for file in onlyfiles[:1000]:
        filesrc = src + file
        imholder = CadastralImage(filesrc, resolution=resolution, mult=mult, device=device)
        dataset.append((imholder.vec, 1))  # because the label of real image is True (i.e. 1)
    
    return dataset
