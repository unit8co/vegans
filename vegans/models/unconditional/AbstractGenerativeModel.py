import os
import sys
import time
import json
import torch

import numpy as np
import vegans.utils.utils as utils
import matplotlib.pyplot as plt

from torch.nn import MSELoss
from datetime import datetime
from abc import ABC, abstractmethod
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from vegans.utils.utils import plot_losses
from torch.utils.tensorboard import SummaryWriter

class AbstractGenerativeModel(ABC):
    """The AbstractGenerativeModel is the most basic building block of vegans. All GAN implementation should
    at least inherit from this class. If a conditional version is implemented look at `AbstractConditionalGenerativeModel`.

    Parameters
    ----------
    x_dim : list, tuple
        Number of the output dimensions of the generator and input dimension of the discriminator / critic.
        In the case of images this will be [nr_channels, nr_height_pixels, nr_width_pixels].
    z_dim : int, list, tuple
        Number of the latent dimensions for the generator input. Might have dimensions of an image.
    optim : dict or torch.optim
        Optimizer used for each network. Could be either an optimizer from torch.optim or a dictionary with network
        name keys and torch.optim as value, i.e. {"Generator": torch.optim.Adam}.
    optim_kwargs : dict
        Optimizer keyword arguments used for each network. Must be a dictionary with network
        name keys and dictionary with keyword arguments as value, i.e. {"Generator": {"lr": 0.0001}}.
    feature_layer : torch.nn.*
        Output layer used to compute the feature loss. Should be from either the discriminator or critic.
        If `feature_layer` is not None, the original generator loss is replaced by a feature loss, introduced
        [here](https://arxiv.org/abs/1606.03498v1).
    fixed_noise_size : int
        Number of images shown when logging. The fixed noise is used to produce the images in the folder/images
        subdirectory, the tensorboard images tab and the samples in get_training_results().
    device : string
        Device used while training the model. Either "cpu" or "cuda".
    ngpu : int
        Number of gpus used during training if device == "cuda".
    folder : string
        Creates a folder in the current working directory with this name. All relevant files like summary, images, models and
        tensorboard output are written there. Existing folders are never overwritten or deleted. If a folder with the same name
        already exists a time stamp is appended to make it unique.
    """

    #########################################################################
    # Actions before training
    #########################################################################
    def __init__(self, x_dim, z_dim, optim, optim_kwargs, feature_layer, fixed_noise_size, device, ngpu, folder, secure):
        self.x_dim = tuple([x_dim]) if isinstance(x_dim, int) else tuple(x_dim)
        self.z_dim = tuple([z_dim]) if isinstance(z_dim, int) else tuple(z_dim)
        self.ngpu = ngpu if ngpu is not None else 0
        self.secure = secure
        self.fixed_noise_size = fixed_noise_size
        self.device = device
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        elif device not in ["cuda", "cpu"]:
            raise ValueError("device must be cuda or cpu.")

        if feature_layer is not None:
            assert isinstance(feature_layer, torch.nn.Module), (
                "`feature_layer` must inherit from nn.Module. Is: {}.".format(type(feature_layer))
            )
        self.feature_layer = feature_layer
        if self.feature_layer is not None:
            self.feature_layer.to(self.device)
        if not hasattr(self, "folder"):
            if folder is None:
                self.folder = folder
            else:
                self.folder = folder
                if os.path.exists(folder):
                    now = datetime.now()
                    now = now.strftime("%Y%m%d_%H%M%S")
                    self.folder = folder + "_" + now
                os.makedirs(self.folder)

        self.loss_functions = self._define_loss()
        self.optimizers = self._define_optimizers(optim=optim, optim_kwargs=optim_kwargs)
        self.to(self.device)

        self.fixed_noise = self.sample(n=fixed_noise_size)
        self._check_attributes()
        self.hyperparameters = {
            "x_dim": x_dim, "z_dim": z_dim, "ngpu": ngpu, "folder": folder, "optimizers": self.optimizers,
            "device": self.device, "loss_functions": self.loss_functions
        }
        self._init_run = True

        if hasattr(self, "generator") and hasattr(self, "adversary"):
            self.TYPE = "GAN"
            self._Z_transformer = self.generator
            self._X_transformer = self.adversary
        elif hasattr(self, "encoder") and hasattr(self, "decoder"):
            self.TYPE = "VAE"
            self._Z_transformer = self.decoder
            self._X_transformer = self.encoder
        else:
            raise NotImplementedError(
                "Model should either have self.generator and self.adversary (self.TYPE = 'GAN') or \n"+
                "self.decoder and self.encoder (self.TYPE = 'VAE')."
            )
        self.images_produced = True if len(self._Z_transformer.output_size) == 3 else False
        self.eval()

    def _define_optimizers(self, optim, optim_kwargs):
        """ Define the optimizers dictionary.

        After this step self.optimizers must exist and be of the form
        {"Network1": optimizer1, "Network2": optimizer2, ...}

        Parameters
        ----------
        optim : torch.optim or dict
            If dict, must be of form {"Network1": torch.optim1, "Network2": torch.optim2, ...}. If one network is
            not specified as a key of the dictionary the default optimizer (self._default_optimizer()) will be
            fetched.
        optim_kwargs : dict
            If dict, must be of form {"Network1": optim_kwargs1, "Network2": optim_kwargs1, ...}. If one network is
            not specified as a key of the dictionary no arguments = {} are passed.
        """
        dict_optim_kwargs = {}
        for name, _ in self.neural_nets.items():
            dict_optim_kwargs[name] = {}
        if isinstance(optim_kwargs, dict):
            for name, opt_kwargs in optim_kwargs.items():
                dict_optim_kwargs[name] = opt_kwargs
        self._check_dict_keys(param_dict=dict_optim_kwargs, where="_define_optimizers_kwargs")

        dict_optim = {}
        if optim is None:
            for name, _ in self.neural_nets.items():
                dict_optim[name] = self._default_optimizer()
        elif isinstance(optim, dict):
            for name, opt in optim.items():
                dict_optim[name] = opt
            for name, _ in self.neural_nets.items():
                if name not in dict_optim:
                    dict_optim[name] = self._default_optimizer()
        else:
            for name, _ in self.neural_nets.items():
                dict_optim[name] = optim
        self._check_dict_keys(param_dict=dict_optim, where="_define_optimizers")

        optimizers = {}
        for name, network in self.neural_nets.items():
            optimizers[name] = dict_optim[name](params=network.parameters(), **dict_optim_kwargs[name])
        return optimizers

    def _default_optimizer(self):
        return torch.optim.Adam

    def _check_dict_keys(self, param_dict, where):
        """ Checks if `param_dict` has the correct form.

        The correct form is {"Network1": value1, "Network2": value2, ...} for every network.
        This utility function checks if every network in `self.neural_nets` has an entry in `param_dict`
        and if every entry in `param_dict` exits in `self.neural_nets`.

        Parameters
        ----------
        param_dict : TYPE
            Parameter dictionary
        where : TYPE
            Specifies the location from which the function is called for meaningful `KeyError`.

        Raises
        ------
        KeyError
            If `param_dict` is not of the specified form.
        """
        for name, _ in param_dict.items():
            if name not in self.neural_nets:
                available_nets = [name for name, _ in self.neural_nets.items()]
                raise KeyError("Error in {}: `{}` not in self.neural_nets. Must be one of: {}.".format(
                    where, name, available_nets)
                )
        for name, _ in self.neural_nets.items():
            if name not in param_dict:
                raise KeyError("Error in {}: self.{} not in param_dict.".format(
                    where, name)
                )

    def _check_attributes(self):
        assert hasattr(self, "neural_nets"), "Model must have attribute 'neural_nets'."
        assert hasattr(self, "device"), "Model must have attribute 'device'."
        assert hasattr(self, "optimizers"), "Model must have attribute 'optimizers'."
        assert isinstance(self.ngpu, int) and self.ngpu >= 0, "ngpu must be positive integer. Given: {}.".format(ngpu)
        assert len(self.z_dim) == 1 or len(self.z_dim) == 3, (
            "z_dim must either have length 1 (for vector input) or 3 (for image input). Given: {}.".format(z_dim)
        )
        assert len(self.x_dim) == 1 or len(self.x_dim) == 3, (
            "x_dim must either have length 1 (for vector input) or 3 (for image input). Given: {}.".format(x_dim)
        )
        assert isinstance(self.neural_nets, dict), "'neural_nets' attribute of AbstractGenerativeModel must be dictionary."
        self._check_dict_keys(self.optimizers, where="_define_optimizer_kwargs")

    @abstractmethod
    def _define_loss(self):
        pass

    def _set_up_training(self, X_train, y_train, X_test, y_test, epochs, batch_size, steps,
        print_every, save_model_every, save_images_every, save_losses_every, enable_tensorboard):
        """ Create the dataloaders, SummaryWriters for tensorboard and transform the saving indicators.

        This function creates all data needed during training like the data loaders and save steps.
        It also creates the hyperparameter dictionary and the `steps` dictionary.
        """
        train_dataloader, test_dataloader = self._set_up_data(
            X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, batch_size=batch_size
        )
        nr_test = 0 if X_test is None else len(X_test)

        writer_train = writer_test = None
        if enable_tensorboard:
            assert self.folder is not None, (
                "`folder` argument in constructor was set to `None`. `enable_tensorboard` must be False or `folder` needs to be specified."
            )
            writer_train = SummaryWriter(os.path.join(self.folder, "tensorboard/train/"))
            if X_test is not None:
                writer_test = SummaryWriter(os.path.join(self.folder, "tensorboard/test/"))

        self.steps = self._create_steps(steps=steps)
        save_periods = self._set_up_saver(
            print_every=print_every, save_model_every=save_model_every, save_images_every=save_images_every,
            save_losses_every=save_losses_every, nr_batches=len(train_dataloader)
        )
        self.hyperparameters.update({
            "epochs": epochs, "batch_size": batch_size, "steps": self.steps,
            "print_every": print_every, "save_model_every": save_model_every, "save_images_every": save_images_every,
            "enable_tensorboard": enable_tensorboard, "nr_train": len(X_train), "nr_test": nr_test
        })
        return train_dataloader, test_dataloader, writer_train, writer_test, save_periods

    def _set_up_data(self, X_train, y_train, X_test, y_test, batch_size):
        """ If `X_train` / `X_test` are not data loaders, create them.

        Also asserts their input shapes for consistency.
        """
        x_train_batch, y_train_batch, x_test_batch, y_test_batch = self._get_batch(
            X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, batch_size=batch_size
        )

        if self.secure:
            self._assert_shapes(X_train=x_train_batch, y_train=y_train_batch, X_test=x_test_batch, y_test=y_test_batch)

        train_dataloader = X_train
        if not isinstance(X_train, DataLoader):
            train_data = utils.DataSet(X=X_train, y=y_train)
            train_dataloader = DataLoader(train_data, batch_size=batch_size)

        test_dataloader = X_test
        if X_test is not None and not isinstance(X_test, DataLoader):
            test_data = utils.DataSet(X=X_test, y=y_test)
            test_dataloader = DataLoader(test_data, batch_size=batch_size)

        return train_dataloader, test_dataloader

    def _get_batch(self, X_train, y_train, X_test, y_test, batch_size):
        """ Returns the first batch of the training and testing data for the consistency checks.
        """
        if isinstance(X_train, DataLoader):
            assert y_train is None, (
                "If `X_train` is of type torch.utils.data.DataLoader, `y_train` must be None. The dataloader must " +
                "return values for X and y when iterating."
            )
            try:
                x_train_batch, y_train_batch = iter(X_train).next()
            except ValueError:
                x_train_batch = iter(X_train).next()
                y_train_batch = None
        else:
            x_train_batch = X_train[:batch_size]
            y_train_batch = y_train[:batch_size] if y_train is not None else None

        if isinstance(X_test, DataLoader):
            assert y_test is None, (
                "If `X_test` is of type torch.utils.data.DataLoader, `y_test` must be None. The dataloader must " +
                "return values for X and y when iterating."
            )
            if X_test is not None:
                try:
                    x_test_batch, y_test_batch = iter(X_test).next()
                except ValueError:
                    x_test_batch = iter(X_test).next()
                    y_test_batch = None
        else:
            x_test_batch = X_test[:batch_size] if X_test is not None else None
            y_test_batch = y_test[:batch_size] if y_test is not None else None

        return x_train_batch, y_train_batch, x_test_batch, y_test_batch

    def _assert_shapes(self, X_train, y_train, X_test, y_test):
        assert len(X_train.shape) == 2 or len(X_train.shape) == 4, (
            "X_train must be either have 2 or 4 shape dimensions. Given: {}.".format(X_train.shape) +
            "Try to use X_train.reshape(-1, 1) or X_train.reshape(-1, 1, height, width)."
        )
        assert X_train.shape[1:] == self.x_dim, (
            "Wrong input shape for adversary / encoder. Given: {}. Needed: {}.".format(X_train.shape, self.x_dim)
        )

        if X_test is not None:
            assert X_train.shape[1:] == X_test.shape[1:], (
                "X_train and X_test must have same dimensions. Given: {} and {}.".format(X_train.shape[1:], X_test.shape[1:])
            )

    def _create_steps(self, steps):
        """ Creates the `self.steps` dictionary.

        The created dictionary must be of the form {"Network1": steps1, "Network2": steps2, ...}. During training
        "Network1" will be trained for `steps1` steps, and so on.
        Functionality similar to `self_define_optimizers()`.

        Parameters
        ----------
        steps : None, dict
            If not None a dictionary of the form {"Network1": steps1, "Network2": steps2, ...} is expected.
            This dictionary might also be partially filled.
        """
        steps = {}
        for name, neural_net in self.neural_nets.items():
            steps[name] = 1
        if steps is not None:
            assert isinstance(steps, dict), "steps parameter must be of type dict. Given: {}.".format(type(steps))
            steps = steps
            for name, _ in self.neural_nets.items():
                if name not in steps:
                    steps[name] = 1
            self._check_dict_keys(steps, where="_create_steps")
        return steps

    def _set_up_saver(self, print_every, save_model_every, save_images_every, save_losses_every, nr_batches):
        """ Calculates saving indicators if strings are passed. Additionally corresponding folders are created
        in `self.folder`.

        Returns
        -------
        Returns all saving indicators as integers.
        """
        print_every = self._string_to_batchnr(log_string=print_every, nr_batches=nr_batches, name="print_every")
        if save_model_every is not None:
            save_model_every = self._string_to_batchnr(log_string=save_model_every, nr_batches=nr_batches, name="save_model_every")
            assert self.folder is not None, (
                "`folder` argument in constructor was set to `None`. `save_model_every` must be None or `folder` needs to be specified."
            )
            os.mkdir(os.path.join(self.folder, "models/"))
        if save_images_every is not None:
            save_images_every = self._string_to_batchnr(log_string=save_images_every, nr_batches=nr_batches, name="save_images_every")
            assert self.folder is not None, (
                "`folder` argument in constructor was set to `None`. `save_images_every` must be None or `folder` needs to be specified."
            )
            os.mkdir(os.path.join(self.folder, "images/"))
        save_losses_every = self._string_to_batchnr(log_string=save_losses_every, nr_batches=nr_batches, name="save_losses_every")
        self.total_training_time = 0
        self.current_timer = time.perf_counter()
        self.batch_training_times = []

        return print_every, save_model_every, save_images_every, save_losses_every

    def _string_to_batchnr(self, log_string, nr_batches, name):
        """ Transforms string of the form "0.2e" into 0.2 and performs basic sanity checks.
        """
        if isinstance(log_string, str):
            assert log_string.endswith("e"), "If `{}` is string, must end with 'e' (for epoch), e.g. 0.25e.".format(name)
            save_epochs = float(log_string.split("e")[0])
            log_string = max([int(save_epochs*nr_batches), 1])
        return log_string



    #########################################################################
    # Actions during training
    #########################################################################
    def fit(self, X_train, X_test=None, epochs=5, batch_size=32, steps=None,
        print_every="1e", save_model_every=None, save_images_every=None, save_losses_every="1e", enable_tensorboard=False):
        """ Trains the model, iterating over all contained networks.

        Parameters
        ----------
        X_train : np.array or torch.utils.data.DataLoader
            Training data for the generative network. Usually images.
        X_test : np.array, optional
            Testing data for the generative network. Must have same shape as X_train.
        epochs : int, optional
            Number of epochs (passes over the training data set) performed during training.
        batch_size : int, optional
            Batch size used when creating the data loader from X_train. Ignored if torch.utils.data.DataLoader is passed
            for X_train.
        steps : dict, optional
            Dictionary with names of the networks to indicate how often they should be trained, i.e. {"Generator": 5} indicates
            that the generator is trained 5 times while all other networks are trained once.
        print_every : int, string, optional
            Indicates after how many batches the losses for the train data should be printed to the console. Can also be a string
            of the form "0.25e" (4 times per epoch), "1e" (once per epoch) or "3e" (every third epoch).
        save_model_every : int, string, optional
            Indicates after how many batches the model should be saved. Can also be a string
            of the form "0.25e" (4 times per epoch), "1e" (once per epoch) or "3e" (every third epoch).
        save_images_every : int, string, optional
            Indicates after how many batches the images for the losses and fixed_noise should be saved. Can also be a string
            of the form "0.25e" (4 times per epoch), "1e" (once per epoch) or "3e" (every third epoch).
        save_losses_every : int, string, optional
            Indicates after how many batches the losses for the train and test data should be calculated. Can also be a string
            of the form "0.25e" (4 times per epoch), "1e" (once per epoch) or "3e" (every third epoch).
        enable_tensorboard : bool, optional
            Flag to indicate whether subdirectory folder/tensorboard should be created to log losses and images.
        """
        if not self._init_run:
            raise ValueError("Run initializer of the AbstractGenerativeModel class is your subclass!")
        train_dataloader, test_dataloader, writer_train, writer_test, save_periods = self._set_up_training(
            X_train, y_train=None, X_test=X_test, y_test=None, epochs=epochs, batch_size=batch_size, steps=steps,
            print_every=print_every, save_model_every=save_model_every, save_images_every=save_images_every,
            save_losses_every=save_losses_every, enable_tensorboard=enable_tensorboard
        )
        max_batches = len(train_dataloader)
        test_x_batch = iter(test_dataloader).next().to(self.device).float() if X_test is not None else None
        print_every, save_model_every, save_images_every, save_losses_every = save_periods
        train_x_batch = iter(train_dataloader).next()
        if len(train_x_batch) != batch_size:
            raise ValueError(
                "Return value from train_dataloader has wrong shape. Should return object of size batch_size. " +
                "Did you pass a dataloader to `X_train` containing labels as well?"
            )

        self.train()
        if save_images_every is not None:
            self._log_images(images=self.generate(z=self.fixed_noise), step=0, writer=writer_train)
        for epoch in range(epochs):
            print("---"*20)
            print("EPOCH:", epoch+1)
            print("---"*20)
            for batch, X in enumerate(train_dataloader):
                batch += 1
                step = epoch*max_batches + batch
                X = X.to(self.device).float()
                Z = self.sample(n=len(X))
                for name, _ in self.neural_nets.items():
                    for _ in range(self.steps[name]):
                        self._losses = self.calculate_losses(X_batch=X, Z_batch=Z, who=name)
                        self._zero_grad(who=name)
                        self._backward(who=name)
                        self._step(who=name)

                if print_every is not None and step % print_every == 0:
                    self._losses = self.calculate_losses(X_batch=X, Z_batch=Z)
                    self._summarise_batch(
                        batch=batch, max_batches=max_batches, epoch=epoch,
                        max_epochs=epochs, print_every=print_every
                    )

                if save_model_every is not None and step % save_model_every == 0:
                    self.save(name="models/model_{}.torch".format(step))

                if save_images_every is not None and step % save_images_every == 0:
                    self._log_images(images=self.generate(z=self.fixed_noise), step=step, writer=writer_train)
                    self._save_losses_plot()

                if save_losses_every is not None and step % save_losses_every == 0:
                    self._log_losses(X_batch=X, Z_batch=Z, mode="Train")
                    if enable_tensorboard:
                        self._log_scalars(step=step, writer=writer_train)
                    if test_x_batch is not None:
                        self._log_losses(X_batch=test_x_batch, Z_batch=self.sample(n=len(test_x_batch)), mode="Test")
                        if enable_tensorboard:
                            self._log_scalars(step=step, writer=writer_test)

        self.eval()
        self._clean_up(writers=[writer_train, writer_test])


    @abstractmethod
    def calculate_losses(self, X_batch, Z_batch, who=None):
        pass

    def _calculate_feature_loss(self, X_real, X_fake):
        """ Calculates feature loss if `self.feature_layer` is not None.

        Every network takes the `feature_layer` argument in its constructor.
        If it is not None, a layer of the discriminator / critic should be specified.
        A feature loss will be calculated which is the MSE between the output for real
        and fake samples of the specified `self.feature_layer`.

        Parameters
        ----------
        X_real : torch.Tensor
            Real samples.
        X_fake : torch.Tensor
            Fake samples.
        """
        X_real_features = self.feature_layer(X_real)
        X_fake_features = self.feature_layer(X_fake)
        feature_loss = MSELoss()(X_real_features, X_fake_features)
        return feature_loss

    def _zero_grad(self, who=None):
        if who is not None:
            self.optimizers[who].zero_grad()
        else:
            [optimizer.zero_grad() for _, optimizer in self.optimizers.items()]

    def _backward(self, who=None):
        assert len(self._losses) != 0, "'self._losses' empty when performing '_backward'."
        if who is not None:
            self._losses[who].backward(retain_graph=True)
        else:
            [loss.backward(retain_graph=True) for _, loss in self._losses.items()]

    def _step(self, who=None):
        if who is not None:
            self.optimizers[who].step()
        else:
            [optimizer.step() for _, optimizer in self.optimizers.items()]


    #########################################################################
    # Logging during training
    #########################################################################
    def _summarise_batch(self, batch, max_batches, epoch, max_epochs, print_every):
        """ Print summary statistics after a specified amount of batches.

        Parameters
        ----------
        batch : int
            Current batch.
        max_batches : int
            Maximum number of total batches.
        epoch : int
            Current epoch.
        max_epochs : int
            Maximum number of total epochs.
        print_every : int
            After how many batches the summary should be printed.
        """
        step = epoch*max_batches + batch
        max_steps = max_epochs*max_batches
        remaining_batches = max_epochs*max_batches - step
        print("Step: {} / {} (Epoch: {} / {}, Batch: {} / {})".format(
            step, max_steps, epoch+1, max_epochs, batch, max_batches)
        )
        print("---"*20)
        for name, loss in self._losses.items():
            print("{}: {}".format(name, loss.item()))

        self.batch_training_times.append(time.perf_counter() - self.current_timer)
        self.total_training_time = np.sum(self.batch_training_times)
        time_per_batch = np.mean(self.batch_training_times) / print_every

        print("\n")
        print("Time left: ~{} minutes (Steps remaining: {}).".format(
            np.round(remaining_batches*time_per_batch/60, 3), remaining_batches
            )
        )
        self.current_timer = time.perf_counter()

    def _log_images(self, images, step, writer):
        if self.images_produced:
            if writer is not None:
                grid = make_grid(images)
                writer.add_image('images', grid, step)

            fig, axs = self._build_images(images)
            if fig is not None:
                plt.savefig(os.path.join(self.folder, "images/image_{}.png".format(step)))
                plt.close()
            print("Images logged.")

    @staticmethod
    def _build_images(images):
        """ Build matplotlib figure containing all images.

        Parameters
        ----------
        images : torch.Tensor()
            A torch Tensor containing the images to be plotted by matplotlib.

        Returns
        -------
        fig, axs: plt.Figure, np.array
            Objects containing the figure as well as all separate axis objects.
        """
        images = images.detach().cpu().numpy()
        if images.shape[1] == 1:
            images = images[:, 0, :, :]
        elif images.shape[1] == 3:
            images = images.reshape((-1, images.shape[2], images.shape[3], 3))
        else:
            return None, None
        nrows = int(np.sqrt(len(images)))
        ncols = len(images) // nrows
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 10))

        for i, (ax, image) in enumerate(zip(np.ravel(axs), images)):
            ax.imshow(image)
            ax.axis("off")
        return fig, axs

    def _log_losses(self, X_batch, Z_batch, mode):
        self._losses = self.calculate_losses(X_batch=X_batch, Z_batch=Z_batch)
        self._append_losses(mode=mode)

    def _append_losses(self, mode):
        if not hasattr(self, "logged_losses"):
            self.logged_losses = self._create_logged_losses()
        for name, loss in self._losses.items():
            self.logged_losses[mode][name].append(self._losses[name].item())

    def _create_logged_losses(self):
        with_test = self.hyperparameters["nr_test"] is not None
        logged_losses = {"Train": {}}
        if with_test:
            logged_losses["Test"] = {}

        for name, _ in self._losses.items():
            logged_losses["Train"][name] = []
            if with_test:
                logged_losses["Test"][name] = []

        return logged_losses

    def _save_losses_plot(self):
        """ Creates the `losses.png` plot in the `self.folder` path.
        """
        if hasattr(self, "logged_losses"):
            fig, axs = plot_losses(self.logged_losses, show=False, share=False)
            plt.savefig(os.path.join(self.folder, "losses.png"))
            plt.close()

    def _log_scalars(self, step, writer):
        """ Log all scalars with tensorboard.
        """
        if writer is not None:
            for name, loss in self._losses.items():
                writer.add_scalar("Loss/{}".format(name), loss.item(), step)
            writer.add_scalar("Time/Total", self.total_training_time / 60, step)
            writer.add_scalar("Time/Batch", np.mean(self.batch_training_times) / 60, step)

    #########################################################################
    # After training
    #########################################################################
    def _clean_up(self, writers=None):
        [writer.close() for writer in writers if writer is not None]

    def get_training_results(self, by_epoch=False, agg=None):
        """ Call after training to get fixed_noise samples and losses.

        Parameters
        ----------
        by_epoch : bool, optional
            If true one loss value per epoch is returned for every logged_loss. Otherwise frequency is given
            by `save_losses_every` argument of `fit`, i.e. `save_losses_every=10` saves losses every 10th batch,
            `save_losses_every="0.25e` saves losses 4 times per epoch.
        agg : None, optional
            Aggregation function used if by_epoch is true, otherwise ignored. Default is np.mean for all batches
            in one epoch.

        Returns
        -------
        samples : np.array
            Images produced with the final model for the fixed_noise.
        losses_dict : dict
            Dictionary containing all loss types logged during training
        """
        samples = self.generate(z=self.fixed_noise)
        losses = self.get_losses(by_epoch=by_epoch, agg=agg)
        return samples, losses

    def get_losses(self, by_epoch=False, agg=None):
        """ Get losses logged during training

        Parameters
        ----------
        by_epoch : bool, optional
            If true one loss value per epoch is returned for every logged_loss. Otherwise frequency is given
            by `save_losses_every` argument of `fit`, i.e. `save_losses_every=10` saves losses every 10th batch,
            `save_losses_every="0.25e` saves losses 4 times per epoch.
        agg : None, optional
            Aggregation function used if by_epoch is true, otherwise ignored. Default is np.mean for all batches
            in one epoch.

        Returns
        -------
        losses_dict : dict
            Dictionary containing all loss types logged during training
        """
        if agg is None:
            agg = np.mean
        assert callable(agg), "agg: Aggregation function must be callable."
        losses_dict = self.logged_losses.copy()
        if by_epoch:
            epochs = self.get_hyperparameters()["epochs"]
            for mode, loss_dict in losses_dict.items():
                for key, losses in loss_dict.items():
                    batches_per_epoch = len(losses) // epochs
                    loss_dict[key] = [losses[epoch*batches_per_epoch:(epoch+1)*batches_per_epoch] for epoch in range(epochs)]
                    loss_dict[key] = [agg(loss_epoch) for loss_epoch in loss_dict[key]]

        return losses_dict


    #########################################################################
    # Saving and loading
    #########################################################################
    def save(self, name=None):
        """ Saves model in the model folder as torch / pickle object.

        Parameters
        ----------
        name : str, optional
            name of the saved file. folder specified in the constructor used
            in absolute path.
        """
        if name is None:
            name = "model.torch"
        torch.save(self, os.path.join(self.folder, name))
        print("Model saved to {}.".format(os.path.join(self.folder, name)))

    @staticmethod
    def load(path):
        """ Load an already trained model.

        Parameters
        ----------
        path : TYPE
            path to the saved file.

        Returns
        -------
        AbstractGenerativeModel
            Trained model
        """
        return torch.load(path)


    #########################################################################
    # Utility functions
    #########################################################################
    def sample(self, n):
        """ Sample from the latent distribution.

        Parameters
        ----------
        n : int
            Number of samples drawn from the latent distribution.

        Returns
        -------
        torch.tensor
            Random numbers with shape of [n, *z_dim]
        """
        return torch.randn(size=(n, *self.z_dim), requires_grad=True, device=self.device)

    def generate(self, z=None, n=None):
        """ Generate output with generator / decoder.

        Parameters
        ----------
        z : None, optional
            Latent input vector to produce an output from.
        n : None, optional
            Number of outputs to be generated.

        Returns
        -------
        np.array
            Output produced by generator / decoder.
        """
        return self(z=z, n=n)

    def predict(self, x):
        """ Use the critic / discriminator to predict if input is real / fake.

        Parameters
        ----------
        x : np.array
            Images or samples to be predicted.

        Returns
        -------
        np.array
            Array with one output per x indicating the realness of an input.
        """
        if not isinstance(x, torch.Tensor):
            x = torch.from_numpy(x).to(self.device)
        predictions = self._X_transformer(x)
        return predictions

    def get_hyperparameters(self):
        """ Returns a dictionary containing all relevant hyperparameters.

        Returns
        -------
        dict
            Dictionary containing all relevant hyperparameters.
        """
        return self.hyperparameters

    def summary(self, save=False):
        """ Print summary of the model in Keras style way.

        Parameters
        ----------
        save : bool, optional
            If true summary is saved in model folder, printed to console otherwise.
        """
        if save:
            assert self.folder is not None, (
                "`folder` argument in constructor was set to `None`. `enable_tensorboard` must be False or `folder` needs to be specified."
            )
            sys_stdout_temp = sys.stdout
            sys.stdout = open(os.path.join(self.folder, 'summary.txt'), 'w')
        for name, neural_net in self.neural_nets.items():
            neural_net.summary()
            print("\n\n")
        print("Hyperparameters\n---------------")
        for key, value in self.get_hyperparameters().items():
            print("{}: ---> {}".format(str(key), str(value)))
        if save:
            sys.stdout = sys_stdout_temp
            sys_stdout_temp

    def get_number_params(self):
        """ Returns the number of parameters in the model.

        Returns
        -------
        dict
            Dictionary containing the number of parameters per network.
        """
        nr_params_dict = {}
        for name, neural_net in self.neural_nets.items():
            nr_params_dict[name] = neural_net.get_number_params()
        return nr_params_dict


    def eval(self):
        """ Set all networks to evaluation mode.
        """
        self.training = False
        [network.eval() for name, network in self.neural_nets.items()]

    def train(self):
        """ Set all networks to training mode.
        """
        self.training = True
        [network.train() for name, network in self.neural_nets.items()]

    def to(self, device):
        """ Map all networks to device.
        """
        [network.to(device) for name, network in self.neural_nets.items()]

    def __call__(self, z=None, n=None):
        if z is not None and n is not None:
            raise ValueError("Only one of 'z' and 'n' is needed.")
        elif z is None and n is None:
            raise ValueError("Either 'z' or 'n' must be not None.")
        elif n is not None:
            z = self.sample(n=n)
        sample = self._Z_transformer(z)
        if self.training:
            return sample
        return sample.detach().cpu().numpy()

    def __str__(self):
        self.summary()