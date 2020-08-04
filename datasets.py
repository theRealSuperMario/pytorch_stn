import torch
from torchvision import datasets
import numpy as np
import os


class MNIST_TRANSLATED(datasets.MNIST):
    def __init__(
        self,
        root,
        random_state=42,
        target_label=8,
        train=True,
        transform=None,
        target_transform=None,
        download=False,
    ):
        super(MNIST_TRANSLATED, self).__init__(
            root=root,
            train=train,
            transform=transform,
            target_transform=target_transform,
            download=download,
        )
        path = "index_list.npz"
        self.index_list = np.load(path)["index_list"]
        offsets = np.random.RandomState(random_state).randint(
            0, 84 ** 2, size=(len(self.index_list),)
        )
        self.index_numbers = np.arange(0, len(self.index_list))
        self.offsets = offsets
        self.target_label = target_label

    def __len__(self):
        target_idxs = self.index_list == self.target_label
        return int(np.sum(target_idxs))

    def __getitem__(self, i):
        valid_idx = self.index_list == self.target_label
        idx = self.index_numbers[valid_idx][i]
        o = self.offsets[idx]
        o = np.unravel_index(o, (84, 84))

        canvas = np.zeros((84 + 28, 84 + 28))
        start_x, start_y = o
        end_x = start_x + 28
        end_y = start_y + 28
        img, label = super(MNIST_TRANSLATED, self).__getitem__(idx)
        canvas[start_x:end_x, start_y:end_y] = img
        return canvas


if __name__ == "__main__":
    dset = datasets.MNIST(root=".", download=True)
    from matplotlib import pyplot as plt

    print("generating index list")
    index_list = []
    for (img, label) in dset:
        index_list.append(label)
    index_list = np.array(index_list)
    np.savez_compressed("index_list.npz", index_list=index_list)

    dset = MNIST_TRANSLATED(root=".", download=True, target_label=8)
    imgs = [dset[i] for i in range(16)]
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    axes = axes.ravel()
    for ax, img in zip(axes, imgs):
        ax.imshow(img)
        ax.set_axis_off()
    fig.suptitle("Translated MNIST digit 8")
    fig.savefig("Translated_mnist_8.png")
    print("iterating over Translated mnist once")
    for _ in dset:
        pass

    dset = MNIST_TRANSLATED(root=".", download=True, target_label=0)
    imgs = [dset[i] for i in range(16)]
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    axes = axes.ravel()
    for ax, img in zip(axes, imgs):
        ax.imshow(img)
        ax.set_axis_off()
    fig.suptitle("Translated MNIST digit 0")
    fig.savefig("Translated_mnist_0.png")
    print("iterating over Translated mnist once")
    for _ in dset:
        pass

