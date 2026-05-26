"""
Actor and critic networks for SAC-based algorithms.
"""

import torch
from torch import nn

from utils.common import SquashedNormal


class Actor(nn.Module):
    """
    Gaussian policy network with tanh squashing.
    """

    def __init__(self, observation_size: int, num_actions: int):
        super().__init__()

        hidden_size = [256, 256]

        self.log_std_bounds = [-20, 2]

        self.act_net = nn.Sequential(
            nn.Linear(observation_size, hidden_size[0]),
            nn.ReLU(),
            nn.Linear(hidden_size[0], hidden_size[1]),
            nn.ReLU(),
        )

        self.mean_linear = nn.Linear(hidden_size[1], num_actions)
        self.log_std_linear = nn.Linear(hidden_size[1], num_actions)

    def forward(
        self,
        state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        x = self.act_net(state)

        mu = self.mean_linear(x)
        log_std = self.log_std_linear(x)

        # Bound log standard deviation
        log_std = torch.tanh(log_std)

        log_std_min, log_std_max = self.log_std_bounds

        log_std = (
            log_std_min
            + 0.5 * (log_std_max - log_std_min) * (log_std + 1)
        )

        std = log_std.exp()

        dist = SquashedNormal(mu, std)

        sample = dist.rsample()
        log_pi = dist.log_prob(sample).sum(-1, keepdim=True)

        return sample, log_pi, dist.mean


class Critic(nn.Module):
    """
    Twin Q-network critic used by SAC.
    """

    def __init__(self, observation_size: int, num_actions: int):
        super().__init__()

        hidden_size = [256, 256]
        input_size = observation_size + num_actions

        self.Q1 = nn.Sequential(
            nn.Linear(input_size, hidden_size[0]),
            nn.ReLU(),
            nn.Linear(hidden_size[0], hidden_size[1]),
            nn.ReLU(),
            nn.Linear(hidden_size[1], 1),
        )

        self.Q2 = nn.Sequential(
            nn.Linear(input_size, hidden_size[0]),
            nn.ReLU(),
            nn.Linear(hidden_size[0], hidden_size[1]),
            nn.ReLU(),
            nn.Linear(hidden_size[1], 1),
        )

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        obs_action = torch.cat([state, action], dim=1)

        q1 = self.Q1(obs_action)
        q2 = self.Q2(obs_action)

        return q1, q2