import os
import torch

import numpy as np
import pandas as pd
import matplotlib as mpl
import utils.utils as utils
import matplotlib.pyplot as plt


def plot_2d_grid(model, nr_images=10, show=True):
    image_dim = model.adversary.input_size[1]
    the_max = 2
    x_limit = np.linspace(-the_max, the_max, nr_images)
    y_limit = np.linspace(the_max, -the_max, nr_images)
    image = np.empty((image_dim*nr_images, image_dim*nr_images))
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(9, 9))

    for j, yi in enumerate(y_limit):
        for i, xi in enumerate(x_limit):
            # z_input = torch.randn(size=(3, *model.generator.input_size), requires_grad=False).to(model.device)
            z_input = torch.Tensor([[xi, yi]]).to(model.device)
            generated_image = model(x=z_input).cpu().detach().numpy()
            image[i*image_dim:(i+1)*image_dim, j*image_dim:(j+1)*image_dim] = generated_image[0, 0, :, :]
    ax.imshow(image, cmap="gray")
    ax.grid(False)
    xticks = ax.get_xticks()[1:-1]
    yticks = ax.get_yticks()[1:-1]
    plt.xticks(xticks, np.round(np.linspace(-the_max, the_max, len(xticks)), 2))
    plt.yticks(yticks, np.round(np.linspace(-the_max, the_max, len(yticks)), 2))
    if show:
        plt.show()
    return fig, ax

def plot_on_click(model):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(9, 9))
    plt.plot([-2, -2, 2, 2, -2], [-2, 2, 2, -2, -2])
    ax.set_title("Latent space")
    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()

def onclick(event):
    z_input = torch.Tensor([[event.xdata, event.ydata]]).to(model.device)
    generated_image = model(x=z_input).cpu().detach().numpy()
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 4))
    ax.imshow(generated_image[0, 0, :, :], cmap="gray", origin=None)
    ax.set_title("(x, y) = ({}, {})".format(round(event.xdata, 2), round(event.ydata, 2)))
    plt.show()

if __name__ == '__main__':
    datapath = "./Data/mnist/"
    model_path = "./TrainedModels/GAN/model.torch"
    model = torch.load(model_path)
    model.eval()
    assert model.generator.input_size[0] == 2, (
        "Wrong input_size required. Given: {}. Needed: 2.".format(model.generator.input_size[0])
    )
    plot_2d_grid(model=model, nr_images=20, show=False)
    plot_on_click(model=model)


