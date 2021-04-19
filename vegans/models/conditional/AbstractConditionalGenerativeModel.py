import torch

import numpy as np
import matplotlib.pyplot as plt
import vegans.utils.utils as utils

from torch.nn import MSELoss
from torchvision.utils import make_grid
from vegans.utils.utils import get_input_dim
from vegans.models.unconditional.AbstractGenerativeModel import AbstractGenerativeModel


class AbstractConditionalGenerativeModel(AbstractGenerativeModel):
    #########################################################################
    # Actions before training
    #########################################################################
    def __init__(self, x_dim, z_dim, y_dim, optim, optim_kwargs, feature_layer, fixed_noise_size, device, folder, ngpu):
        """The AbstractConditionalGenerativeModel is the most basic building block of VeGAN for conditional models. All conditional GAN
        implementation should at least inherit from this class.

        Parameters
        ----------
        x_dim : list, tuple
            Number of the output dimension of the generator and input dimension of the discriminator / critic.
            In the case of images this will be [nr_channels, nr_height_pixels, nr_width_pixels].
        z_dim : int, list, tuple
            Number of the latent dimension for the generator input. Might have dimensions of an image.
        y_dim : int, list, tuple
            Number of dimensions for the target label. Might have dimensions of image for image to image translation, i.e.
            [nr_channels, nr_height_pixels, nr_width_pixels] or an integer
        optim : dict or torch.optim
            Optimizer used for each network. Could be either an optimizer from torch.optim or a dictionary with network
            name keys and torch.optim as value, i.e. {"Generator": torch.optim.Adam}.
        optim_kwargs : dict
            Optimizer keyword arguments used for each network. Must be a dictionary with network
            name keys and dictionary with keyword arguments as value, i.e. {"Generator": {"lr": 0.0001}}.
        feature_layer : torch.nn.*
            Output layer used to compute the feature loss. Should be from either the discriminator or critic.
            If `feature_layer` is not None, the original generator loss is replaced by a feature loss.
        fixed_noise_size : int
            Number of images shown when logging. The fixed noise is used to produce the images in the folder/images
            subdirectory, the tensorboard images tab and the samples in get_training_results().
        device : string
            Device used while training the model. Either "cpu" or "cuda".
        folder : string
            Creates a folder in the current working directory with this name. All relevant files like summary, images, models and
            tensorboard output are written there. Existing folders are never overwritten or deleted. If a folder with the same name
            already exists a time stamp is appended to make it unique.
        ngpu : int
            Number of gpus used during training if device == "cuda".
        """
        self.adv_in_dim = get_input_dim(dim1=x_dim, dim2=y_dim)
        self.gen_in_dim = get_input_dim(dim1=z_dim, dim2=y_dim)
        AbstractGenerativeModel.__init__(
            self, x_dim=x_dim, z_dim=z_dim, optim=optim, optim_kwargs=optim_kwargs, feature_layer=feature_layer,
            fixed_noise_size=fixed_noise_size, device=device, folder=folder, ngpu=ngpu
        )
        self.y_dim = tuple([y_dim]) if isinstance(y_dim, int) else y_dim
        self.hyperparameters["y_dim"] = self.y_dim
        self.eval()

    def _set_up_training(self, X_train, y_train, X_test, y_test, epochs, batch_size, steps,
        print_every, save_model_every, save_images_every, save_losses_every, enable_tensorboard):
        train_dataloader, test_dataloader, writer_train, writer_test, save_periods = super()._set_up_training(
            X_train, y_train, X_test, y_test, epochs, batch_size, steps,
            print_every, save_model_every, save_images_every, save_losses_every, enable_tensorboard
        )
        self.fixed_labels = y_train[:self.fixed_noise_size]
        self.fixed_labels = torch.from_numpy(self.fixed_labels).to(self.device)
        return train_dataloader, test_dataloader, writer_train, writer_test, save_periods

    def _assert_shapes(self, X_train, y_train, X_test, y_test):
        assert len(X_train.shape) == 2 or len(X_train.shape) == 4, (
            "X_train must be either have 2 or 4 shape dimensions. Given: {}.".format(X_train.shape) +
            "Try to use X_train.reshape(-1, 1) or X_train.reshape(-1, 1, height, width)."
        )
        assert X_train.shape[1:] == self.x_dim[:], (
            "Wrong input shape for adversariat / encoder. Given: {}. Needed: {}.".format(X_train.shape, self.x_dim)
        )

        if X_test is not None:
            assert X_train.shape[1:] == X_test.shape[1:], (
                "X_train and X_test must have same dimensions. Given: {} and {}.".format(X_train.shape[1:], X_test.shape[1:])
            )
            if y_test is not None:
                assert X_test.shape[0] == y_test.shape[0], (
                    "Same number if X_test and y_test needed.Given: {} and {}.".format(X_test.shape[0], y_test.shape[0])
                )

        assert X_train.shape[0] == y_train.shape[0], (
            "Same number if X_train and y_train needed.Given: {} and {}.".format(X_train.shape[0], y_train.shape[0])
        )
        assert len(y_train.shape) == 2 or len(y_train.shape) == 4, (
            "y_train must be either have 2 or 4 shape dimensions. Given: {}.".format(y_train.shape) +
            "Try to use y_train.reshape(-1, 1) or y_train.reshape(-1, 1, height, width)."
        )
        assert y_train.shape[2:] == self.y_dim[1:], (
            "Wrong input shape for y_train. Given: {}. Needed: {}.".format(y_train.shape, self.y_dim)
        )
        if len(y_train.shape) == 4:
            assert X_train.shape == y_train.shape, (
                "If y_train is an image (Image-to-Image translation task) it must have the same shape as X_train" +
                " to be concatenated before passing it to the discriminator. x_shape: {}. y_shape: {}.".format(X_train.shape, y_train.shape)
            )
        if X_test is not None:
            assert y_train.shape[1:] == y_test.shape[1:], (
                "y_train and y_test must have same dimensions. Given: {} and {}.".format(y_train.shape[1:], y_test.shape[1:])
            )

    @staticmethod
    def _check_conditional_network_input(network, in_dim, y_dim, name):
        from vegans.utils.networks import NeuralNetwork
        arch = NeuralNetwork._get_iterative_layers(network, input_type="Object")
        has_error = False

        for i, layer in enumerate(arch):
            if "in_features" in layer.__dict__:
                inpt_layer = layer
                inpt_dim = int(layer.__dict__["in_features"])
                if inpt_dim == in_dim:
                    has_error = True
                elif np.prod(inpt_dim) == np.prod(in_dim):
                    has_error = True
                break
            elif "in_channels" in layer.__dict__:
                inpt_layer = layer
                inpt_dim = int(layer.__dict__["in_channels"])
                if inpt_dim == in_dim[0]:
                    has_error = True
                break
        else:
            raise TypeError("No input layer found. No Linear or Conv2d /ConvTranspose2d layers?")

        if has_error:
            required_dim = get_input_dim(in_dim, y_dim)
            if len(required_dim) == 1:
                first_layer = "torch.nn.Linear(in_features={}, out_features=...)".format(required_dim[0])

            else:
                first_layer = (
                    "torch.nn.Conv2d(in_channels={}, out_channels=...)\n".format(required_dim[0]) +
                    "\t\t\t\t\t\t\t\t\ttorch.nn.ConvTranspose2d(in_channels={}, out_channels=...)\n".format(required_dim[0]) +
                    "\t\t\t\t\t\t\t\t\ttorch.nn.Linear(in_features={}, out_features=...) if torch.nn.Flatten() is used before".format(np.prod(required_dim))
                )
            raise AssertionError(
                "\n\n**{}** is a conditional network.".format(name) +
                "The y_dim (label) will be concatenated to the input of this network.\n\n" +
                "The first layer will receive input shape: {} due to y_dim={}. ".format(required_dim, y_dim) +
                "Given: {}.(Reshape & Flatten not considered)\n".format(str(layer)) +
                "First layer should be of the form: {}.\n\n".format(first_layer) +
                "Please use vegans.utils.utils.get_input_dim(in_dim, y_dim) to get the correct input dimensions.\n" +
                "Check on github for notebooks of conditional GANs.\n\n"
            )

    @staticmethod
    def _check_unconditional_network_input(network, in_dim, y_dim, name):
        from vegans.utils.networks import NeuralNetwork
        arch = NeuralNetwork._get_iterative_layers(network, input_type="Object")
        has_error = False
        concat_input = get_input_dim(in_dim, y_dim)

        for i, layer in enumerate(arch):
            if "in_features" in layer.__dict__:
                inpt_layer = layer
                inpt_dim = int(layer.__dict__["in_features"])
                if inpt_dim == concat_input:
                    has_error = True
                elif np.prod(inpt_dim) == np.prod(concat_input):
                    has_error = True
                break
            elif "in_channels" in layer.__dict__:
                inpt_layer = layer
                inpt_dim = int(layer.__dict__["in_channels"])
                if inpt_dim == concat_input[0]:
                    has_error = True
                break
        else:
            raise TypeError("No input layer found. No Linear or Conv2d layers?")

        if has_error:
            if len(in_dim) == 1:
                first_layer = "torch.nn.Linear(in_features={}, out_features=...)".format(in_dim[0])

            else:
                first_layer = (
                    "torch.nn.Conv2d(in_channels={}, out_channels=...)\n".format(in_dim[0]) +
                    "\t\t\t\t\t\t\t\t\ttorch.nn.ConvTranspose2d(in_channels={}, out_channels=...)\n".format(in_dim[0]) +
                    "\t\t\t\t\t\t\t\t\ttorch.nn.Linear(in_features={}, out_features=...) if torch.nn.Flatten() is used before".format(np.prod(in_dim))
                )
            raise AssertionError(
                "\n\n**{}** is **not** a conditional network. The y_dim (label) will **not** be concatenated to the input of this network.\n\n".format(name) +
                "The first layer will receive input shape: {} (same as x_dim). Given: {}.(Reshape & Flatten not considered)\n".format(in_dim, str(layer)) +
                "First layer should be of the form: {}.\n\n".format(first_layer) +
                "Please use vegans.utils.utils.get_input_dim(in_dim, y_dim) to get the correct input dimensions.\n" +
                "Check on github for notebooks of conditional GANs.\n\n"
            )

    #########################################################################
    # Actions during training
    #########################################################################
    def fit(self, X_train, y_train, X_test=None, y_test=None, epochs=5, batch_size=32, steps=None,
            print_every="1e", save_model_every=None, save_images_every=None, save_losses_every="1e", enable_tensorboard=True):
        """ Method to call when the conditional generative adversarial network should be trained.

        Parameters
        ----------
        X_train : np.array
            Training data for the generative adversarial network. Usually images.
        y_train: np.array
            Training labels for the generative adversarial network. Might be images or one-hot encoded vector.
        X_test : np.array, optional
            Testing data for the generative adversarial network. Must have same shape as X_train.
        y_train: np.array
            Testing labels for the generative adversarial network. Might be images or one-hot encoded vector.
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
        train_dataloader, test_dataloader, writer_train, writer_test, save_periods = self._set_up_training(
            X_train, y_train, X_test=X_test, y_test=y_test, epochs=epochs, batch_size=batch_size, steps=steps,
            print_every=print_every, save_model_every=save_model_every, save_images_every=save_images_every,
            save_losses_every=save_losses_every, enable_tensorboard=enable_tensorboard
        )
        max_batches = len(train_dataloader)
        test_x_batch = next(iter(test_dataloader))[0].to(self.device) if X_test is not None else None
        test_y_batch = next(iter(test_dataloader))[1].to(self.device) if X_test is not None else None
        print_every, save_model_every, save_images_every, save_losses_every = save_periods

        self.train()
        if save_images_every is not None:
            self._log_images(
                images=self.generate(y=self.fixed_labels, z=self.fixed_noise),
                step=0, writer=writer_train
            )

        for epoch in range(epochs):
            print("---"*20)
            print("EPOCH:", epoch+1)
            print("---"*20)
            for batch, (X, y) in enumerate(train_dataloader):
                batch += 1
                step = epoch*max_batches + batch
                X = X.to(self.device).float()
                y = y.to(self.device)
                Z = self.sample(n=len(X))
                for name, _ in self.neural_nets.items():
                    for _ in range(self.steps[name]):
                        self._losses = {}
                        self.calculate_losses(X_batch=X, Z_batch=Z, y_batch=y, who=name)
                        self._zero_grad(who=name)
                        self._backward(who=name)
                        self._step(who=name)

                if print_every is not None and step % print_every == 0:
                    self._losses = {}
                    self.calculate_losses(X_batch=X, Z_batch=Z, y_batch=y)
                    self._summarise_batch(
                        batch=batch, max_batches=max_batches, epoch=epoch,
                        max_epochs=epochs, print_every=print_every
                    )

                if save_model_every is not None and step % save_model_every == 0:
                    self.save(name="models/model_{}.torch".format(step))

                if save_images_every is not None and step % save_images_every == 0:
                    self._log_images(
                        images=self.generate(y=self.fixed_labels, z=self.fixed_noise),
                        step=step, writer=writer_train
                    )
                    self._save_losses_plot()

                if save_losses_every is not None and step % save_losses_every == 0:
                    self._log_losses(X_batch=X, Z_batch=Z, y_batch=y, mode="Train")
                    if enable_tensorboard:
                        self._log_scalars(step=step, writer=writer_train)
                    if test_x_batch is not None:
                        self._log_losses(
                            X_batch=test_x_batch, Z_batch=self.sample(n=len(test_x_batch)),
                            y_batch=test_y_batch, mode="Test"
                        )
                        if enable_tensorboard:
                            self._log_scalars(step=step, writer=writer_test)

        self.eval()
        self._clean_up(writers=[writer_train, writer_test])

    def _calculate_feature_loss(self, X_real, X_fake, y_batch):
        X_real = utils.concatenate(tensor1=X_real, tensor2=y_batch).float()
        X_fake = utils.concatenate(tensor1=X_fake, tensor2=y_batch).float()

        X_real_features = self.feature_layer(X_real)
        X_fake_features = self.feature_layer(X_fake)
        feature_loss = MSELoss()(X_real_features, X_fake_features)
        return feature_loss


    #########################################################################
    # Logging during training
    #########################################################################
    def _log_images(self, images, step, writer):
        assert len(self.x_dim) > 1, (
            "Called _log_images in AbstractGenerativeModel for adversariat / encoder.input_size = {}.".format(self.x_dim)
        )
        if writer is not None:
            grid = make_grid(images)
            writer.add_image('images', grid, step)

        fig, axs = self._build_images(images)
        for i, ax in enumerate(np.ravel(axs)):
            try:
                lbl = torch.argmax(self.fixed_labels[i], axis=0).item()
                ax.set_title("Label: {}".format(lbl))
            except ValueError:
                pass
        plt.savefig(self.folder+"images/image_{}.png".format(step))
        plt.close()
        print("Images logged.")

    def _log_losses(self, X_batch, Z_batch, y_batch, mode):
        self._losses = {}
        self.calculate_losses(X_batch=X_batch, Z_batch=Z_batch, y_batch=y_batch)
        self._append_losses(mode=mode)


    #########################################################################
    # After training
    #########################################################################
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
        losses_dict : dict
            Dictionary containing all loss types logged during training
        """
        samples = self.generate(y=self.fixed_labels, z=self.fixed_noise)
        losses = self.get_losses(by_epoch=by_epoch, agg=agg)
        return samples, losses

    def get_fixed_labels(self):
        return self.fixed_labels.cpu().numpy()


    #########################################################################
    # Utility functions
    #########################################################################
    def concatenate(self, tensor1, tensor2):
        return utils.concatenate(tensor1=tensor1, tensor2=tensor2)

    def generate(self, y, z=None):
        """ Generate output with generator.

        Parameters
        ----------
        y : np.array
            Labels for outputs to be produced.
        z : None, optional
            Latent input vector to produce an output from.

        Returns
        -------
        np.array
            Output produced by generator.
        """
        return self(y=y, z=z)

    def predict(self, x, y):
        """ Use the critic / discriminator to predict if input is real / fake.

        Parameters
        ----------
        x : np.array
            Images or samples to be predicted.
        y : np.array
            Labels for outputs to be predicted.

        Returns
        -------
        np.array
            Array with one output per x indicating the realness of an input.
        """
        inpt = self.concatenate(x, y).float().to(self.device)
        return self._X_transformer(inpt)

    def __call__(self, y, z=None):
        if z is None:
            z = self.sample(n=len(y))

        if not isinstance(y, torch.Tensor):
            y = torch.from_numpy(y).to(self.device)
        inpt = self.concatenate(z, y).float()
        sample = self._Z_transformer(inpt)
        if self._is_training:
            return sample
        return sample.detach().cpu().numpy()
