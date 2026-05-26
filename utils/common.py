"""
Common neural network utilities and probability distributions.

This module contains reusable components shared across multiple RL algorithms,
including SAC, REDQ, TQC, and future actor-critic methods.
"""

from torch import distributions as pyd
from torch import nn
from torch.distributions.transformed_distribution import (
    TransformedDistribution,
)
from torch.distributions.transforms import TanhTransform
from torch.nn import functional as F


class MLP(nn.Module):
    """
    Standard multilayer perceptron (MLP).

    Args:
        input_size (int):
            Input feature dimension.

        hidden_sizes (list[int]):
            Hidden layer dimensions.

        output_size (int):
            Output feature dimension.
    """

    def __init__(
        self,
        input_size: int,
        hidden_sizes: list[int],
        output_size: int,
    ):
        super().__init__()

        self.fully_connected_layers = []

        for i, next_size in enumerate(hidden_sizes):

            fully_connected_layer = nn.Linear(input_size, next_size)

            self.add_module(
                f"fully_connected_layer_{i}",
                fully_connected_layer,
            )

            self.fully_connected_layers.append(
                fully_connected_layer
            )

            input_size = next_size

        self.output_layer = nn.Linear(
            input_size,
            output_size,
        )

    def forward(self, state):

        for fully_connected_layer in self.fully_connected_layers:
            state = F.relu(fully_connected_layer(state))

        output = self.output_layer(state)

        return output


class StableTanhTransform(TanhTransform):
    """
    Numerically stable tanh transform.

    Overrides inverse tanh to reduce NaN instability issues that may occur
    during SAC log-probability calculations.
    """

    def __init__(self, cache_size=1):
        super().__init__(cache_size=cache_size)

    @staticmethod
    def atanh(x):
        return 0.5 * (x.log1p() - (-x).log1p())

    def __eq__(self, other):
        return isinstance(other, StableTanhTransform)

    def _inverse(self, y):
        """
        Inverse tanh transform.

        Clamping is intentionally avoided to preserve gradient behaviour.
        """
        return self.atanh(y)


# These methods are intentionally ignored for SAC usage.
# pylint: disable=abstract-method
class SquashedNormal(TransformedDistribution):
    """
    Gaussian distribution followed by tanh squashing.

    Commonly used in SAC-style stochastic policies to produce
    bounded continuous actions.
    """

    def __init__(self, loc, scale):

        self.loc = loc
        self.scale = scale

        self.base_dist = pyd.Normal(loc, scale)

        transforms = [StableTanhTransform()]

        super().__init__(
            self.base_dist,
            transforms,
            validate_args=False,
        )

    @property
    def mean(self):
        """
        Returns the transformed distribution mean.
        """

        mu = self.loc

        for transform in self.transforms:
            mu = transform(mu)

        return mu