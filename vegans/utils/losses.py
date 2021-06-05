import torch

class KLLoss():
    def __init__(self, eps):
        self.eps = eps

    def __call__(self, input, target):
        """ Compute the Kullback-Leibler loss for GANs.

        Parameters
        ----------
        input : torch.Tensor
            Input tensor. Output of a critic.

        Returns
        -------
        torch.Tensor
            KL divergence
        """
        return -torch.mean(torch.log(input / (1 + self.eps - input) + self.eps))

class WassersteinLoss():
    def __call__(self, input, target):
        """ Compute the Wasserstein loss / divergence.

        Also known as earthmover distance.

        Parameters
        ----------
        input : torch.Tensor
            Input tensor. Output of a critic.
        target : torch.Tensor
            Label, either 1 or -1. Zeros are translated to -1.

        Returns
        -------
        torch.Tensor
            Wasserstein divergence
        """
        assert torch.unique(target).shape[0] <= 2, "Only two different values for target allowed."
        target[target==0] = -1

        return torch.mean(target*input)

class NormalNegativeLogLikelihood():
    def __call__(self, x, mu, variance, eps=1e-6):
        negative_log_likelihood = 1/(2*variance + eps)*(x-mu)**2 + 0.5*torch.log(variance + eps)
        negative_log_likelihood = negative_log_likelihood.sum(axis=1).mean()
        return negative_log_likelihood