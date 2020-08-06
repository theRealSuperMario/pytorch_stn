#
# Spatial Transformer Networks Tutorial
# =====================================
# **Author**: `Ghassen HAMROUNI <https://github.com/GHamrouni>`_
# With modificationy by `Sandro Braun`
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
from nets import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def normalize(x):
    return (x - 0.1307) / 0.3081


dset_train = MNIST_TRANSLATED(
    root=".",
    train=True,
    download=True,
    transform=transforms.Compose([transforms.ToTensor()]),
)
dset_train_pairs = StochasticPairs(dset_train, random_state=42)
dset_test = MNIST_TRANSLATED(
    root=".", train=True, transform=transforms.Compose([transforms.ToTensor()])
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


model_encoder = EncoderNet().to(device)
model_encoder.load_state_dict(torch.load("classifier_pretrained.pth"))
model_encoder.train()
model_stn = LocalizationNet().to(device)

random_crop = torchvision.transforms.RandomCrop([28, 28])

params = list(model_encoder.parameters()) + list(model_stn.parameters())
optimizer = optim.SGD(params, lr=0.01)


def train(epoch):
    model_encoder.train()
    model_stn.train()
    for batch_idx, (I_n, I_m) in enumerate(train_loader):
        I_n = I_n.to(device)
        I_m = I_m.to(device)
        I_n = normalize(I_n)
        I_m = normalize(I_m)

        I_n_crop = model_stn(I_n)
        I_m_crop = model_stn(I_m)
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


def convert_image_np(inp):
    """Convert a Tensor to numpy image."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    return inp


def visualize_stn(epoch):
    with torch.no_grad():
        model_stn.eval()
        # Get a batch of training data
        I_n = next(iter(test_loader))[0].to(device)
        in_grid = (
            torchvision.utils.make_grid(I_n).detach().permute(1, 2, 0).cpu().numpy()
        )

        thetas_n = model_stn.stn_theta(normalize(I_n)).cpu().numpy()
        image_grid = []
        W = 84 + 28
        H = 84 + 28
        print(thetas_n[0, ...])

        for theta_n, img_n in zip(thetas_n, I_n):
            img_n = img_n.detach().cpu()
            img = img_n.numpy().transpose((1, 2, 0))
            img = np.concatenate([img] * 3, -1)
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
                cv2.polylines(img_with_box, polygons, True, (0, 255, 255), 3) / 255.0
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
        f.suptitle(f"{epoch}.png")
        axarr[0].imshow(in_grid)
        axarr[0].set_title("Dataset Images")

        axarr[1].imshow(image_grid)
        axarr[1].set_title("Learned Transformations")

        f.tight_layout()


for epoch in range(1, 100):
    train(epoch)
    if epoch % 1 == 0:
        visualize_stn(epoch)
        import os

        os.makedirs("colocalization", exist_ok=True)
        plt.savefig(f"colocalization/{epoch:06d}.png")
        plt.close()

# Visualize the STN transformation on some input batch

plt.ioff()
plt.show()

