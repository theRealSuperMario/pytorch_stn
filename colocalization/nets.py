from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from datasets import MNIST_TRANSLATED, StochasticPairs
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from utils import *
import cv2


class LocalizationNet(nn.Module):
    def __init__(self):
        super(LocalizationNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=7, stride=2)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=7, stride=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(1600, 512)
        self.fc2 = nn.Linear(512, 64)
        self.fc3 = nn.Linear(64, 8)
        self.fc4 = nn.Linear(8, 3)

        self.fc4.weight.data.zero_()
        self.fc4.bias.data.copy_(
            torch.tensor([1.0, 0.0, 0.0], dtype=torch.float)
        )  # scale, tx, ty

        # Spatial transformer localization-network
        self.localization_conv = nn.Sequential(
            self.conv1, self.pool1, nn.ReLU(True), self.conv2, self.pool2,
        )
        self.localization_fc = nn.Sequential(
            self.fc1,
            nn.ReLU(True),
            self.fc2,
            nn.ReLU(True),
            self.fc3,
            nn.ReLU(True),
            self.fc4,
        )

    # Spatial transformer network forward function
    def stn(self, x):
        theta = self.stn_theta(x)

        grid = F.affine_grid(theta, (x.size()[0], 1, 28, 28))  # TODO: check this
        x = F.grid_sample(x, grid)
        return x

    def stn_theta(self, x):
        y = self.localization_conv(x)
        y = nn.Flatten()(y)
        theta_params = self.localization_fc(y)
        theta = torch.zeros((theta_params.shape[0], 2, 3), dtype=theta_params.dtype).to(
            theta_params.device
        )
        theta[:, 0, 0] = theta_params[:, 0]  # scale
        theta[:, 1, 1] = theta_params[:, 0]  # scale
        theta[:, 0, 2] = theta_params[:, 1]  # scale
        theta[:, 1, 2] = theta_params[:, 2]  # scale
        return theta

    def forward(self, x):
        # transform the input
        x = self.stn(x)
        return x


class EncoderNet(nn.Module):
    def __init__(self):
        super(EncoderNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 3 * 3, 32), nn.ReLU(True), nn.Linear(32, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(
            torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float)
        )

    # Spatial transformer network forward function
    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 3 * 3)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x

    def stn_theta(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 3 * 3)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        return theta

    def encode(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        e1 = x.view(-1, 320)
        e2 = self.fc1(e1)
        x = F.relu(e2)
        e3 = self.fc2(x)
        embedding = torch.cat([e1, e2, e3], dim=1)
        embedding = embedding / embedding.norm(dim=1, keepdim=True)
        return embedding

    def forward(self, x):
        # transform the input
        x = self.stn(x)

        # Perform the usual forward pass
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
