#
# Spatial Transformer Networks Tutorial
# =====================================
# **Author**: `Ghassen HAMROUNI <https://github.com/GHamrouni>`_
#
# .. figure:: /_static/img/stn/FSeq.png
#
# In this tutorial, you will learn how to augment your network using
# a visual attention mechanism called spatial transformer
# networks. You can read more about the spatial transformer
# networks in the `DeepMind paper <https://arxiv.org/abs/1506.02025>`__
#
# Spatial transformer networks are a generalization of differentiable
# attention to any spatial transformation. Spatial transformer networks
# (STN for short) allow a neural network to learn how to perform spatial
# transformations on the input image in order to enhance the geometric
# invariance of the model.
# For example, it can crop a region of interest, scale and correct
# the orientation of an image. It can be a useful mechanism because CNNs
# are not invariant to rotation and scale and more general affine
# transformations.
#
# One of the best things about STN is the ability to simply plug it into
# any existing CNN with very little modification.

# License: BSD
# Author: Sandro Braun

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


"""
Quote from the Paper:


We use a margin α = 1, 
and for the encoding function e() we use the CNN trained for digit classification from Sect. 4.1, 
concatenating the three layers of activations (two hidden layers and the classification layer without softmax) to form a feature descriptor. 

We use a spatial trans- former parameterised for attention (scale and translation) where the localisation network is a 100k parameter CNN consisting of a convolutional layer with eight 9 × 9 filters and a 4 pixel stride, followed by 2 × 2 max pooling with stride 2 and then two 8-unit fully-connected layers before the final 3-unit fully-connected layer.

Triplet encoder:
CNN as trained for digit classifcation
* concatenate two hidden layers and classification layer without softmax
"""


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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dset_train = MNIST_TRANSLATED(
    root=".",
    train=True,
    download=True,
    transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    ),
)
dset_train_pairs = StochasticPairs(dset_train, random_state=42)
dset_test = MNIST_TRANSLATED(
    root=".",
    train=True,
    transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    ),
)
dset_test_pairs = StochasticPairs(dset_test, random_state=42)


# Training dataset
train_loader = torch.utils.data.DataLoader(
    dset_train_pairs, batch_size=64, shuffle=True, num_workers=4
)
train_iter = iter(train_loader)
# Test dataset
test_loader = torch.utils.data.DataLoader(
    dset_test_pairs, batch_size=64, shuffle=False, num_workers=4
)


# Depicting spatial transformer networks
# --------------------------------------
#
# Spatial transformer networks boils down to three main components :
#
# -  The localization network is a regular CNN which regresses the
#    transformation parameters. The transformation is never learned
#    explicitly from this dataset, instead the network learns automatically
#    the spatial transformations that enhances the global accuracy.
# -  The grid generator generates a grid of coordinates in the input
#    image corresponding to each pixel from the output image.
# -  The sampler uses the parameters of the transformation and applies
#    it to the input image.
#
# .. figure:: /_static/img/stn/stn-arch.png
#
# .. Note::
#    We need the latest version of PyTorch that contains
#    affine_grid and grid_sample modules.

"""This snippet is from https://github.com/wuneng/WarpAffine2GridSample/pull/4"""


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


model_encoder = EncoderNet().to(device)
model_encoder.load_state_dict(torch.load("classifier_pretrained.pth"))
model_encoder.train()
model_stn = LocalizationNet().to(device)


# Training the model
# ------------------
#
# Now, let's use the SGD algorithm to train the model. The network is
# learning the classification task in a supervised way. In the same time
# the model is learning STN automatically in an end-to-end fashion.

random_crop = torchvision.transforms.RandomCrop([28, 28])

params = list(model_encoder.parameters()) + list(model_stn.parameters())
optimizer = optim.SGD(params, lr=0.01)


def train(epoch):
    model_encoder.train()
    model_stn.train()
    for batch_idx, (I_n, I_m) in enumerate(train_loader):
        I_n = I_n.to(device)
        I_m = I_m.to(device)

        I_n_crop = model_stn(I_n)
        I_m_crop = model_stn(I_m)
        # I_n_randomcrop = random_crop(I_m)
        crop_pos = np.random.randint(0, 84 ** 2 - 1, size=(1,))
        top, left = np.unravel_index(crop_pos, (84, 84))
        height = 28
        width = 28
        I_n_randomcrop = transforms.functional.crop(
            I_n, int(top), int(left), height, width
        )

        e_i_n_crop = model_encoder.encode(I_n_crop)
        e_i_m_crop = model_encoder.encode(I_m_crop)
        e_i_n_randomcrop = model_encoder.encode(I_n_randomcrop)

        alpha = 1.0

        distances = torch.sum((e_i_n_crop - e_i_m_crop) ** 2, dim=1) - torch.sum(
            (e_i_n_crop - e_i_n_randomcrop) ** 2, dim=1
        )
        distances = distances + alpha
        loss = torch.mean(F.relu(distances, inplace=False))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch_idx % 500 == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(dset_train),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )


# Visualizing the STN results
# ---------------------------
#
# Now, we will inspect the results of our learned visual attention
# mechanism.
#
# We define a small helper function in order to visualize the
# transformations while training.
#


def convert_image_np(inp):
    """Convert a Tensor to numpy image."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    return inp


# We want to visualize the output of the spatial transformers layer
# after the training, we visualize a batch of input images and
# the corresponding transformed batch using STN.


def visualize_stn():
    with torch.no_grad():
        model_stn.eval()
        # Get a batch of training data
        I_n = next(iter(test_loader))[0].to(device)
        in_grid = (
            torchvision.utils.make_grid(I_n).detach().permute(1, 2, 0).cpu().numpy()
        )

        thetas_n = model_stn.stn_theta(I_n).cpu().numpy()
        image_grid = []
        W = 84 + 28
        H = 84 + 28
        print(thetas_n[0, ...])

        for theta_n, img_n in zip(thetas_n, I_n):
            img_n = img_n.detach().cpu()
            img = convert_image_np(img_n)
            T = cvt_ThetaToM(theta_n, W, H, return_inv=False)
            T = np.concatenate([T, np.zeros((1, 3))], axis=0).astype(np.float32)
            T[-1, -1] = 1
            points = np.array(
                [[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]], dtype=np.float32
            ).reshape((-1, 1, 2))
            bbox = cv2.perspectiveTransform(points, T)
            bbox = bbox.reshape((-1, 2))

            polygons = [np.int32(bbox)]

            img_with_box = img.copy() * 255.0
            img_with_box = (
                cv2.polylines(img_with_box, polygons, True, (0, 255, 255), 1) / 255.0
            )
            image_grid.append(img_with_box)
        image_grid = np.stack(image_grid, axis=0)
        image_grid = (
            torchvision.utils.make_grid(
                torch.from_numpy(image_grid).permute(0, 3, 1, 2)
            )
            .permute(1, 2, 0)
            .numpy()
        )

        # Plot the results side-by-side
        f, axarr = plt.subplots(1, 2)
        axarr[0].imshow(in_grid)
        axarr[0].set_title("Dataset Images")

        axarr[1].imshow(image_grid)
        axarr[1].set_title("Learned Transformations")

        f.tight_layout()


for epoch in range(1, 100):
    train(epoch)
    if epoch % 1 == 0:
        visualize_stn()
        plt.savefig(f"colocalization/{epoch:06d}.png")
        plt.close()

# Visualize the STN transformation on some input batch
visualize_stn()

plt.ioff()
plt.show()

