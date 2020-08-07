# Script partially from
# https://github.com/skaae/recurrent-spatial-transformer-code/blob/master/MNIST_SEQUENCE/create_mnist_sequence.py

__author__ = "sorensonderby, enhanced by Sandro Braun"
import torch
from torchvision import datasets
import numpy as np
import os


GRID_SIZE = [84, 84]  #   size of grid where MNIST digits will be placed randomly
ORG_SHP = [28, 28]
dist_size = [6, 6]
NUM_DISTORTIONS = 16  #   number of distractors per image to add
distractor_size = 6  #   size of distractors to end.
outfile = "mnist_cluttered_60"
NUM_DISTORTIONS_DB = 100000

PRNG = np.random.RandomState(42)


mnist_data_train = datasets.MNIST(".", train=True, download=True)
mnist_data_test = datasets.MNIST(".", train=False, download=True)


"""Collect set of random distortions"""


distortions = []
for i in range(NUM_DISTORTIONS_DB):
    rand_digit = np.random.randint(num_digits)
    rand_x = np.random.randint(ORG_SHP[1] - dist_size[1])
    rand_y = np.random.randint(ORG_SHP[0] - dist_size[0])

    digit = all_digits[rand_digit]
    distortion = digit[rand_y : rand_y + dist_size[0], rand_x : rand_x + dist_size[1]]
    distortions += [distortion]


dataset = mnist_data_train
X_train, Y_train = [], []
for img, target in dataset:
    canvas = np.zeros((GRID_SIZE[0] + 28, GRID_SIZE[1] + 28, 1))
    location = PRNG.randint(0, np.prod(GRID_SIZE) - 1)
    start_x, start_y = np.unravel_index(location, GRID_SIZE)
    end_x = start_x + 28
    end_y = start_y + 28
    canvas[start_x:end_x, start_y:end_y] = img

    distractor_grid = (
        GRID_SIZE[0] + 28 - distractor_size,
        GRID_SIZE[1] + 28 - distractor_size,
    )
    distractor_canvas = np.zeros((distractor_grid[0], distractor_grid[1], 1))
    num_locations = np.prod(distractor_grid)
    distractor_locations = PRNG.randint(0, num_locations - 1)
    start_x, start_y = np.unravel_index(location, distractor_grid)
    end_x = start_x + distractor_size
    end_y = start_y + distractor_size

    distractor_canvas[start_x:end_x, start_y:end_y] = img
