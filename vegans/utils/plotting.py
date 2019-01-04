import matplotlib.pyplot as plt
import torchvision.utils as vutils
import numpy as np


def plot_losses(G_losses, D_losses):
    """
    Plots losses for generator and discriminator on a common plot,
    even when the two networks have been trained a different number of times.
    :param G_losses:
    :param D_losses:
    :return:
    """

    # First, build the time axis, which is the union of the two list of steps
    time_axis = sorted(set(G_losses.keys()) | set(D_losses.keys()))

    G_vals, D_vals = [], []
    last_G, last_D = float('nan'), float('nan')
    for t in time_axis:
        if t in G_losses:
            last_G = G_losses[t]
        G_vals.append(last_G)
        if t in D_losses:
            last_D = D_losses[t]
        D_vals.append(last_D)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))
    fig.subplots_adjust(hspace=0.4)

    ax1.plot(range(len(time_axis)), G_vals, lw=2)
    ax2.plot(range(len(time_axis)), D_vals, lw=2)
    ax1.set_xlabel('Iterations')
    ax2.set_xlabel('Iterations')
    ax1.set_ylabel('G Loss')
    ax2.set_ylabel('D Loss')
    ax1.set_title('Generator loss')
    ax2.set_title('Discriminator loss')

    plt.show()


def plot_image_samples(image_samples, max_images=20):
    """
    Plots images when samples are images, of dimension (N, C, H, W), where N is the number of samples (at each step),
    C is the number of channels, H is the image height and W the width.
    :param image_samples: the samples to plot
    :param max_images: maximum number of image grids (steps) to plot
    """

    time_steps = sorted(image_samples.keys())

    if len(time_steps) > max_images:
        print('More than {} images to plot ({}); aborting. Use max_images parameter to correct behavior'
              .format(max_images, len(time_steps)))
        return

    for t in time_steps:
        img = image_samples[t]
        plt.figure(figsize=(8, 8))
        plt.axis("off")
        plt.imshow(np.transpose(vutils.make_grid(img, padding=2, normalize=True), (1, 2, 0)))
        plt.title('epoch {} / step {}'.format(*t))

    plt.show()
