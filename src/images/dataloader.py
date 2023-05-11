import torch
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
    def __init__(self, src: str, resolution: Optional[tuple] = None, device: Optional[str] = 'cpu'):
        self.src = src
        self._load_transform(resolution, device)
        
    def _load_transform(self, resolution: Optional[tuple] = None, device: Optional[str] = 'cpu'):
        """
        Loads and transforms the image into 1-D channel representation (with resize if necessary)
        """
        self.image = Image.open(self.src).convert('L')
        self.vec = transforms.functional.to_tensor(self.image).to(device) # "grayscale" convertion from RGB
        # case image is too big we'll resize it according to the resolution provided
        if resolution and resolution != self.image.size:
            if resolution[0] > self.image.size[0] or resolution[1] > self.image.size[1]:
                raise WrongResolutionException(f"Cannot turn image of size {self.image.size} into {resolution}")
            else:
                resizer = transforms.Resize(resolution)
                self.vec = resizer.forward(self.vec)
        # leaving only first dimension after converting
        self.vec = self.vec.reshape(self.vec.shape[1], self.vec.shape[2])
        # leaving only 0.5 value in case of non-binary element
        self.vec = torch.where((self.vec == 0) + (self.vec == 1), self.vec, 0.5)
        
    def show_image(self):
        """
        Showing the image.
        
        NB! If you want to see the original image (not resized), call `self.image` itself in a command line
        """
        plt.imshow(self.vec.cpu())
