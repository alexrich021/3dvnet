import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable
from torch import Tensor
import cv2
import math
from mv3d.baselines.gpmvs.utils import freeze_batchnorm


class GPlayer(nn.Module):
    def __init__(self):
        super(GPlayer, self).__init__()

        self.gamma2 = nn.Parameter(torch.randn(1), requires_grad=True).float()
        self.ell = nn.Parameter(torch.randn(1), requires_grad=True).float()
        self.sigma2 = nn.Parameter(torch.randn(1), requires_grad=True).float()

    def forward(self, D, Y):
        """
        :param D: Distance matrix
        :param Y: Stacked outputs from encoder
        :return: Z: transformed latent space
        """
        # Support for these operations on Half precision is low at the moment, handle everything in Float precision
        batch, latents, channel, height, width = Y.size()
        Y = Y.view(batch, latents, -1).float()
        D = D.float()

        # MATERN CLASS OF COVARIANCE FUNCTION
        # ell > 0, gamma2 > 0, sigma2 > 0 : EXPONENTIATE THEM !!!
        K = torch.exp(self.gamma2) * (1 + math.sqrt(3) * D / torch.exp(self.ell)) * torch.exp(-math.sqrt(3) * D / torch.exp(self.ell))
        I = torch.eye(latents, device=Y.device, dtype=torch.float32).expand(batch, latents, latents)
        C = K + torch.exp(self.sigma2) * I
        Cinv = C.inverse()
        Z = K.bmm(Cinv).bmm(Y)
        Z = torch.nn.functional.relu(Z)
        return Z

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        """
        super(GPlayer, self).train(mode)
        self.apply(freeze_batchnorm)

