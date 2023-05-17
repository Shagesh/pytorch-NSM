"""Define a convolutional NSM module."""

import math

import numpy as np

import torch
import torch.nn.functional as F
from torch import nn

import matplotlib.pyplot as plt


class NSM_Conv(nn.Module):
    def __init__(
        self,
        input_dim,
        encoding_dim,
        k_size,
        etaW=1e-3,
        etaM=1e-3,
        tau=1e-1,
        dropout_p=None,
        x_whitening=False,
    ):
        super(NSM_Conv, self).__init__()
        ## encoder ##
        self.in_channels = input_dim
        self.encoding_dim = encoding_dim
        self.encoder = nn.Conv2d(input_dim, encoding_dim, k_size, stride=1, padding=0)
        torch.nn.init.normal_(self.encoder.weight, mean=0.0, std=1.0)
        for iter_norm in range(encoding_dim):
            self.encoder.weight[iter_norm].data /= torch.linalg.norm(
                self.encoder.weight[iter_norm].detach()
            )

        ## competitor ##
        self.competitor = nn.Linear(encoding_dim, encoding_dim, bias=False)
        self.competitor.weight.data.copy_(torch.eye(encoding_dim))  # /encoding_dim)
        self.dropout_p = dropout_p

        ## dropout and input whitening ##
        if dropout_p is not None:
            self.dropout = nn.Dropout(p=self.dropout_p)
        self.x_whitening = x_whitening

        ## learning rates
        self.etaW = etaW
        self.etaM = etaM
        self.tau = tau

        # Decay/schedule
        self.t_step = 0
        self.decay = 1e-1

    def forward(self, x):
        num_iter = 40
        Wx = self.encoder(x).detach()
        if self.dropout_p is not None:
            Wx = (1 / self.dropout_p) * self.dropout(Wx)

        device = x.device
        M = self.competitor.weight.detach() - torch.eye(
            self.competitor.weight.detach().size(0)
        ).to(device)

        u = torch.zeros(Wx.size()).to(device)
        y = torch.zeros(Wx.size()).to(device)

        for iter_updates in range(num_iter):
            My = torch.einsum("ij,kjxy->kixy", M, y)
            # delta_u = - u + Wx - My
            u = Wx - My
            y = F.relu(u)

        # if self.dropout is not None:
        #  y = self.dropout(y)

        return y.detach()

    def pool_output(self, y, k_size, stri):
        m = nn.MaxPool2d(kernel_size=k_size, stride=stri)
        return m(y).detach()

    def pool_quadrant(self, y):
        Width_Im = y.shape[2]

        if Width_Im % 2 == 0:
            kernel_size = Width_Im // 2
            stride = kernel_size
        else:
            kernel_size = math.ceil(Width_Im / 2)
            stride = kernel_size - 1

        m = nn.MaxPool2d(kernel_size=kernel_size, stride=stride)
        return m(y).detach()

    def loss_NSM_conv(self, y, x, print_=False):
        scaling_factor = y.shape[0] * y.shape[2] * y.shape[3]

        Wx = self.encoder(x)
        YWX = (y * Wx).sum() / scaling_factor

        My = torch.einsum("ij,kjxy->kixy", self.competitor.weight, y)
        YMY = (y * My).sum() / scaling_factor

        if self.x_whitening == True:
            W_reg = (Wx**2).sum() / scaling_factor
        else:
            W_reg = (self.encoder.weight**2).sum()

        M_reg = (self.competitor.weight**2).sum()
        total_loss = -4 * YWX + 2 * YMY + 2 * W_reg - M_reg
        if print_ == True:
            print(
                W_reg.detach().cpu(),
                M_reg.detach().cpu(),
                YWX.detach().cpu(),
                YMY.detach().cpu(),
            )
        return total_loss

    def train(self, y, x, etaW=None, etaM=None, tau=None):
        if etaW is None:
            etaW = self.etaW
        if etaM is None:
            etaM = self.etaM
        if tau is None:
            tau = self.tau

        self.loss_NSM_conv(y, x).backward()

        # gradients
        gW = self.encoder.weight.grad.data
        gM = self.competitor.weight.grad.data

        # updates
        lr_schedule = 1  # 1 / ( 1 + self.t_step * self.decay )

        with torch.no_grad():
            self.encoder.weight -= etaW * lr_schedule * gW
            self.competitor.weight += (etaM / tau) * lr_schedule * gM

        self.encoder.weight.grad.zero_()
        self.competitor.weight.grad.zero_()
        self.t_step += 10

    def plot_features(self):
        # function to plot features

        if self.in_channels == 1:
            k_size = self.encoder.kernel_size[0]
            plt.figure(figsize=(10, 10))
            for iter_filter in range(min(25, self.encoding_dim)):
                plt.subplot(5, 5, iter_filter + 1)
                plt.imshow(
                    self.encoder.weight[iter_filter]
                    .reshape(self.in_channels, k_size, k_size)
                    .detach()
                    .cpu()
                )
            plt.colorbar()
            plt.show()

        else:
            filters = self.encoder.weight.detach().numpy()
            filters_min = np.amin(filters)
            filters_max = np.amax(filters)

            fig, ax = plt.subplots(
                int(np.sqrt(self.encoding_dim)),
                int(np.sqrt(self.encoding_dim)),
                sharex=True,
                sharey=True,
                figsize=(8, 8),
            )
            for i in range(int(np.sqrt(self.encoding_dim))):
                for j in range(int(np.sqrt(self.encoding_dim))):
                    ax[i][j].imshow(
                        (
                            filters[i * int(np.sqrt(self.encoding_dim)) + j, 0]
                            - filters_min
                        )
                        / (filters_max - filters_min)
                    )
            # show the plot
            plt.show()
