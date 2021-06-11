import numpy as np
import matplotlib.pyplot as plt

def plot_losses(losses, show=True, share=False):
    """
    Plots losses for generator and discriminator on a common plot.

    Parameters
    ----------
    losses : dict
        Dictionary containing the losses for some networks. The structure of the dictionary is:
        ```
        {
            mode1: {loss_type1_1: losses1_1, loss_type1_2: losses1_2, ...},
            mode2: {loss_type2_1: losses2_1, loss_type2_2: losses2_2, ...},
            ...
        }
        ```
        where `mode` is probably one of "Train" or "Test", loss_type might be "Generator", "Adversary", "Encoder", ...
        and losses are lists of loss values collected during training.
    show : bool, optional
        If True, `plt.show` is called to visualise the images directly.
    share : bool, optional
        If true, axis ticks are shared between plots.

    Returns
    -------
    plt.figure, plt.axis
        Created figure and axis objects.
    """
    if share:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        for mode, loss_dict in losses.items():
            for loss_type, loss in loss_dict.items():
                ax.plot(loss, lw=2, label=mode+loss_type)
        ax.set_xlabel('Iterations')
        ax.legend()
    else:
        n = len(losses["Train"])
        nrows = int(np.sqrt(n))
        ncols = n // nrows
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 9))
        axs = np.ravel(axs)
        for mode, loss_dict in losses.items():
            for ax, (loss_type, loss) in zip(axs, loss_dict.items()):
                ax.plot(loss, lw=2, label=mode)
                ax.set_xlabel('Iterations')
                ax.set_title(loss_type)
                ax.set_facecolor("#ecffe7")
                ax.legend()
    fig.tight_layout()
    if show:
        plt.show()
    return fig, ax


def plot_images(images, labels=None, show=True, n=None):
    """ Plot a number of input images with optional label

    Parameters
    ----------
    images : np.array
        Must be of shape [nr_samples, height, width] or [nr_samples, height, width, 3].
    labels : np.array, optional
        Array of labels used in the title.
    show : bool, optional
        If True, `plt.show` is called to visualise the images directly.
    n : None, optional
        Number of images to be drawn, maximum is 36.

    Returns
    -------
    plt.figure, plt.axis
        Created figure and axis objects.
    """
    if len(images.shape)==4 and images.shape[1] == 3:
        images = invert_channel_order(images=images)
    elif len(images.shape)==4 and images.shape[1] == 1:
        images = images.reshape((-1, images.shape[2], images.shape[3]))
    if n is None:
        n = images.shape[0]
    if n > 36:
        n = 36
    nrows = int(np.sqrt(n))
    ncols = n // nrows
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(8, 8))
    axs = np.ravel(axs)

    for i, ax in enumerate(axs):
        ax.imshow(images[i])
        ax.axis("off")
        if labels is not None:
            ax.set_title("Label: {}".format(labels[i]))

    fig.tight_layout()
    if show:
        plt.show()
    return fig, axs

def create_gif(source_path, target_path=None):
    """Create a GIF from images contained on the source path.

    Parameters
    ----------
    source_path : string
        Path pointing to the source directory with .png files.
    target_path : string, optional
        Name of the created GIF.
    """
    import os
    import imageio
    source_path = source_path+"/" if not source_path.endswith("/") else source_path
    images = []
    for file_name in sorted(os.listdir(source_path)):
        if file_name.endswith('.png'):
            file_path = os.path.join(source_path, file_name)
            images.append(imageio.imread(file_path))

    if target_path is None:
        target_path = source_path+"movie.gif"
    imageio.mimsave(target_path, images)