import torch.nn as nn
import torch.nn.functional as F
from torch.fft import fftn

class FT(nn.Module):
    def __init__(self):
        """A starter class for fourier transform.

        source: https://pytorch.org/docs/stable/fft.html

        """
        super().__init__()

    def forward(self, X, states=None):

        return x