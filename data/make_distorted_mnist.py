# Script partially from
# https://github.com/skaae/recurrent-spatial-transformer-code/blob/master/MNIST_SEQUENCE/create_mnist_sequence.py

__author__ = "sorensonderby, enhanced by Sandro Braun"
#%%
import numpy as np
from PIL import Image
import scipy.io as sio
import matplotlib
import matplotlib.pyplot as plt
import os
import torch
from torchvision import datasets


#%%
ORG_SHP = [28, 28]
OUT_SHP = [84, 84]
NUM_DISTORTIONS = 16
dist_size = (6, 6)  # should be odd?
NUM_DISTORTIONS_DB = 100000

mnist_data_train = datasets.MNIST(".", train=True, download=True)
mnist_data_test = datasets.MNIST(".", train=False, download=True)

X_train = mnist_data_train.data
Y_train = mnist_data_train.targets
X_test = mnist_data_test.data
Y_test = mnist_data_test.targets

mnist_data = {
    "Y_train": Y_train,
    "X_train": X_train,
    "Y_test": Y_test,
    "X_test": X_test,
}

#%%
outfile = "mnist_cluttered_84"

np.random.seed(1234)

### create list with distortions
all_digits = np.concatenate([X_train, X_test], axis=0)
all_digits = all_digits.reshape([-1] + ORG_SHP)
num_digits = all_digits.shape[0]

distortions = []
for i in range(NUM_DISTORTIONS_DB):
    rand_digit = np.random.randint(num_digits)
    rand_x = np.random.randint(ORG_SHP[1] - dist_size[1])
    rand_y = np.random.randint(ORG_SHP[0] - dist_size[0])

    digit = all_digits[rand_digit]
    distortion = digit[rand_y : rand_y + dist_size[0], rand_x : rand_x + dist_size[1]]
    # assert distortion.shape == dist_size
    # plt.imshow(distortion, cmap='gray')
    # plt.show()
    distortions += [distortion]

#%%
def shift_digit(x):
    canvas = np.zeros((OUT_SHP[0], OUT_SHP[1]))
    PRNG = np.random.RandomState(42)
    start_x = np.random.randint(0, OUT_SHP[0] - ORG_SHP[0])
    start_y = np.random.randint(0, OUT_SHP[1] - ORG_SHP[1])
    end_x = start_x + 28
    end_y = start_y + 28
    canvas[start_x:end_x, start_y:end_y] = x
    return canvas


def sample_digits(n, x):
    """sample n digits from x"""
    n_samples = x.shape[0]
    idxs = np.random.choice(range(n_samples), replace=True, size=n)
    return [x[i] for i in idxs]


def add_distortions(digits, num_distortions):
    canvas = np.zeros_like(digits)
    for i in range(num_distortions):
        rand_distortion = distortions[np.random.randint(NUM_DISTORTIONS_DB)]
        rand_x = np.random.randint(OUT_SHP[1] - dist_size[1])
        rand_y = np.random.randint(OUT_SHP[0] - dist_size[0])
        canvas[
            rand_y : rand_y + dist_size[0], rand_x : rand_x + dist_size[1]
        ] = rand_distortion
    canvas += digits
    return np.clip(canvas, 0, 1)


def create_dataset(X, Y, org_shp, out_shp):
    X_out, Y_out = [], []
    for x, y in zip(X, Y):
        x_in_canvas = shift_digit(x)
        x_in_canvas = add_distortions(x_in_canvas, NUM_DISTORTIONS)
        X_out.append(x_in_canvas)
        Y_out.append(y)

    X_out = np.stack(X_out, axis=0).astype("float32")
    Y_out = np.stack(Y_out, axis=0).astype("int32")
    return X_out, Y_out


#%%
X_train_distorted, y_train_distorted = create_dataset(
    mnist_data["X_train"], mnist_data["Y_train"], ORG_SHP, OUT_SHP
)
X_test_distorted, y_test_distorted = create_dataset(
    mnist_data["X_test"], mnist_data["Y_test"], ORG_SHP, OUT_SHP
)
np.savez_compressed(
    outfile,
    X_train=X_train_distorted,
    y_train=y_train_distorted,
    X_test=X_test_distorted,
    y_test=y_test_distorted,
)

fig, axes = plt.subplots(4, 4, figsize=(6, 6))
axes = axes.ravel()
for x, y, ax in zip(X_train_distorted[:16, ...], y_train_distorted[:16, ...], axes):
    ax.imshow(x)
    ax.set_axis_off()
    ax.set_title(str(y))
plt.savefig(outfile + "_demo.png")
