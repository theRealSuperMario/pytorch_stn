from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import cv2
from lib import utils, nets
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import os

logdir = "classification_experiment_1"
plt.style.use("seaborn-whitegrid")
writer = SummaryWriter(log_dir=logdir)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NB_RC_PARAMS = {
    "figure.figsize": [5, 4],
    "figure.dpi": 150,
    "figure.autolayout": True,
    "legend.frameon": True,
    "axes.titlesize": "xx-large",
    "axes.labelsize": "x-large",
    "xtick.labelsize": "x-large",
    "ytick.labelsize": "x-large",
    "legend.fontsize": "x-large",
}


data_path = "data/mnist_cluttered_60.npz"
data = np.load(data_path)
X_train = torch.Tensor(data["X_train"])
Y_train = torch.Tensor(data["y_train"]).type(torch.LongTensor)
X_test = torch.Tensor(data["X_test"])
Y_test = torch.Tensor(data["y_test"]).type(torch.LongTensor)

X_shape = X_train.shape[1:]
H, W = X_shape

X_train = X_train.view(-1, 1, H, W)
X_test = X_test.view(-1, 1, H, W)

# Tensors are now shaped [N, 1, 40, 40]

train_dataset = TensorDataset(X_train, Y_train)
test_dataset = TensorDataset(X_test, Y_test)
# import pudb

# pudb.set_trace()

# Training dataset
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=64, shuffle=True, num_workers=4,
)
# Test dataset
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=64, shuffle=False, num_workers=4,
)

model = nets.LocalizationNet_60().to(device)

print(f"Using device {device}")

optimizer = optim.SGD(model.parameters(), lr=0.01)


def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        data = normalize(data)

        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 500 == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )
            writer.add_scalar("train/loss", loss, epoch)


def test():
    with torch.no_grad():
        model.eval()
        test_loss = 0
        correct = 0
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            data = normalize(data)
            output = model(data)

            # sum up batch loss
            test_loss += F.nll_loss(output, target, size_average=False).item()
            # get the index of the max log-probability
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        print(
            "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
                test_loss,
                correct,
                len(test_loader.dataset),
                100.0 * correct / len(test_loader.dataset),
            )
        )
        writer.add_scalar("test/loss", test_loss, epoch)
        writer.add_scalar("test/accuracy", correct, epoch)


def denormalize(inp):
    mean = np.array([0.1307])
    std = np.array([0.3081])
    # mean = np.array([0])
    # std = np.array([1])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    return inp


def normalize(x):
    mean = np.array([0.1307], dtype=np.float32)
    std = np.array([0.3081], dtype=np.float32)
    # mean = np.array([0], dtype=np.float32)
    # std = np.array([1], dtype=np.float32)
    mean = torch.from_numpy(mean).view((1, -1, 1, 1))
    std = torch.from_numpy(std).view((1, -1, 1, 1))
    mean = mean.to(x.device)
    std = std.to(x.device)
    x = (x - mean) / std
    return x


def convert_image_np(inp):
    """Convert a Tensor to numpy image."""
    inp = inp.numpy().transpose((1, 2, 0))
    inp = denormalize(inp)
    return inp


def transpose_image_np(inp):
    inp = inp.numpy().transpose((1, 2, 0))
    return inp


def visualize_stn(epoch):
    with torch.no_grad():
        # Get a batch of training data
        data = next(iter(test_loader))[0].to(device)
        data_normalized = normalize(data)

        input_tensor = data.cpu()
        transformed_input_tensor = model.stn(data_normalized).cpu()

        in_grid = convert_image_np(torchvision.utils.make_grid(normalize(input_tensor)))

        out_grid = convert_image_np(
            torchvision.utils.make_grid(transformed_input_tensor)
        )

        thetas = model.stn_theta(data_normalized).cpu().numpy()
        print("Theta:")
        print(thetas[0, ...])
        image_grid = []
        for theta, img in zip(thetas, input_tensor):
            img = transpose_image_np(img)
            T = utils.cvt_ThetaToM(theta, H, W, return_inv=True)
            T = np.concatenate([T, np.zeros((1, 3))], axis=0).astype(np.float32)
            T[-1, -1] = 1
            points = np.array(
                [[0, 0], [H, 0], [H, W], [0, W]], dtype=np.float32
            ).reshape((-1, 1, 2))
            bbox = cv2.perspectiveTransform(points, T)
            bbox = bbox.reshape((-1, 2))

            polygons = [np.int32(bbox)]

            img_with_box = img.copy() * 255.0
            img_with_box = np.concatenate([img_with_box,] * 3, axis=-1)
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
        with plt.rc_context(NB_RC_PARAMS):
            f, axarr = plt.subplots(1, 3, figsize=(18, 8))
            f.suptitle(f"epoch {epoch:03d}", fontsize=26, y=1.02)
            axarr[0].imshow(in_grid)
            axarr[0].set_title("Dataset Images")
            axarr[0].set_axis_off()

            axarr[1].imshow(image_grid)
            axarr[1].set_title("Learned Transformations")
            axarr[1].set_axis_off()

            axarr[2].imshow(out_grid)
            axarr[2].set_title("Transformed Images")
            axarr[2].set_axis_off()

            f.tight_layout()
            plt.savefig(f"{logdir}/{epoch:06d}.png")


for epoch in range(1, 100 + 1):
    train(epoch)
    test()
    if epoch % 5 == 0:
        os.makedirs(logdir, exist_ok=True)
        visualize_stn(epoch)
        torch.save(
            model.state_dict(), f"{logdir}/classifier_weights_{epoch:03d}.pth",
        )
        plt.close()
