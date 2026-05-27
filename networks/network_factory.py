"""
Factory for creating algorithm networks.
"""

import torch

from algorithms.base.sac import SAC
from algorithms.base.td3 import TD3

from algorithms.repetition.resac import ReSAC
from algorithms.repetition.retd3 import ReTD3

from algorithms.sil.sil_sac import SACSIL
from algorithms.sil.sil_td3 import TD3SIL

from networks.sac_networks import (
    Actor as SACActor,
    Critic as SACCritic,
)

from networks.td3_networks import (
    Actor as TD3Actor,
    Critic as TD3Critic,
)


class NetworkFactory:

    @staticmethod
    def create(
        algorithm_name: str,
        observation_size: int,
        action_num: int,
        config,
    ):

        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # =========================================================
        # TD3
        # =========================================================

        if algorithm_name == "TD3":

            actor = TD3Actor(
                observation_size,
                action_num,
            )

            critic = TD3Critic(
                observation_size,
                action_num,
            )

            return TD3(
                actor_network=actor,
                critic_network=critic,
                gamma=config.gamma,
                tau=config.tau,
                action_num=action_num,
                actor_lr=config.actor_lr,
                critic_lr=config.critic_lr,
                device=device,
            )

        # =========================================================
        # ReTD3
        # =========================================================

        if algorithm_name == "ReTD3":

            actor = TD3Actor(
                observation_size,
                action_num,
            )

            critic = TD3Critic(
                observation_size,
                action_num,
            )

            return ReTD3(
                actor_network=actor,
                critic_network=critic,
                gamma=config.gamma,
                tau=config.tau,
                action_num=action_num,
                actor_lr=config.actor_lr,
                critic_lr=config.critic_lr,
                device=device,
            )

        # =========================================================
        # TD3SIL
        # =========================================================

        if algorithm_name == "TD3SIL":

            actor = TD3Actor(
                observation_size,
                action_num,
            )

            critic = TD3Critic(
                observation_size,
                action_num,
            )

            return TD3SIL(
                actor_network=actor,
                critic_network=critic,
                gamma=config.gamma,
                tau=config.tau,
                action_num=action_num,
                actor_lr=config.actor_lr,
                critic_lr=config.critic_lr,
                device=device,
            )

        # =========================================================
        # SAC
        # =========================================================

        if algorithm_name == "SAC":

            actor = SACActor(
                observation_size,
                action_num,
            )

            critic = SACCritic(
                observation_size,
                action_num,
            )

            return SAC(
                actor_network=actor,
                critic_network=critic,
                gamma=config.gamma,
                tau=config.tau,
                reward_scale=config.reward_scale,
                action_num=action_num,
                actor_lr=config.actor_lr,
                critic_lr=config.critic_lr,
                alpha_lr=config.alpha_lr,
                device=device,
            )

        # =========================================================
        # ReSAC
        # =========================================================

        if algorithm_name == "ReSAC":

            actor = SACActor(
                observation_size,
                action_num,
            )

            critic = SACCritic(
                observation_size,
                action_num,
            )

            return ReSAC(
                actor_network=actor,
                critic_network=critic,
                gamma=config.gamma,
                tau=config.tau,
                reward_scale=config.reward_scale,
                action_num=action_num,
                actor_lr=config.actor_lr,
                critic_lr=config.critic_lr,
                alpha_lr=config.alpha_lr,
                device=device,
            )

        # =========================================================
        # SILSAC
        # =========================================================

        if algorithm_name == "SACSIL":

            actor = SACActor(
                observation_size,
                action_num,
            )

            critic = SACCritic(
                observation_size,
                action_num,
            )

            return SACSIL(
                actor_network=actor,
                critic_network=critic,
                gamma=config.gamma,
                tau=config.tau,
                reward_scale=config.reward_scale,
                action_num=action_num,
                actor_lr=config.actor_lr,
                critic_lr=config.critic_lr,
                alpha_lr=config.alpha_lr,
                device=device,
            )

        raise ValueError(
            f"Unknown algorithm: {algorithm_name}"
        )
