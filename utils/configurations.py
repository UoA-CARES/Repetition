"""
Configuration classes for Repetition-RL.
"""

from typing import List, Optional

from pydantic import BaseModel


class SubscriptableClass(BaseModel):
    """Pydantic config with dictionary-style access."""

    def __getitem__(self, item):
        return getattr(self, item)


class EnvironmentConfig(SubscriptableClass):
    task: str


class GymEnvironmentConfig(EnvironmentConfig):
    domain: str = ""
    gym: str = ""
    task: str = ""


class TrainingConfig(SubscriptableClass):
    seeds: List[int] = [10]
    plot_frequency: Optional[int] = 100
    checkpoint_frequency: Optional[int] = 100
    number_steps_per_evaluation: Optional[int] = 10000
    number_eval_episodes: Optional[int] = 10


class AlgorithmConfig(SubscriptableClass):
    algorithm: str

    G: Optional[int] = 1
    buffer_size: Optional[int] = 1000000
    batch_size: Optional[int] = 256

    max_steps_exploration: Optional[int] = 1000
    max_steps_training: Optional[int] = 1000000
    number_steps_per_train_policy: Optional[int] = 1

    min_noise: Optional[float] = 0.0
    noise_scale: Optional[float] = 0.1
    noise_decay: Optional[float] = 1.0


class TD3Config(AlgorithmConfig):
    algorithm: str = "TD3"

    actor_lr: Optional[float] = 3e-4
    critic_lr: Optional[float] = 3e-4

    gamma: Optional[float] = 0.99
    tau: Optional[float] = 0.005


class SACConfig(AlgorithmConfig):
    algorithm: str = "SAC"

    actor_lr: Optional[float] = 3e-4
    critic_lr: Optional[float] = 3e-4
    alpha_lr: Optional[float] = 3e-4

    gamma: Optional[float] = 0.99
    tau: Optional[float] = 0.005
    reward_scale: Optional[float] = 1.0


class ReTD3Config(AlgorithmConfig):
    algorithm: str = "ReTD3"

    actor_lr: Optional[float] = 3e-4
    critic_lr: Optional[float] = 3e-4

    gamma: Optional[float] = 0.99
    tau: Optional[float] = 0.005


class ReSACConfig(AlgorithmConfig):
    algorithm: str = "ReSAC"

    actor_lr: Optional[float] = 3e-4
    critic_lr: Optional[float] = 3e-4
    alpha_lr: Optional[float] = 3e-4

    gamma: Optional[float] = 0.99
    tau: Optional[float] = 0.005
    reward_scale: Optional[float] = 1.0


class TD3SILConfig(AlgorithmConfig):
    algorithm: str = "TD3SIL"

    actor_lr: Optional[float] = 3e-4
    critic_lr: Optional[float] = 3e-4

    gamma: Optional[float] = 0.99
    tau: Optional[float] = 0.005

    sil_weight: Optional[float] = 1.0


class SACSILConfig(AlgorithmConfig):
    algorithm: str = "SACSIL"

    actor_lr: Optional[float] = 3e-4
    critic_lr: Optional[float] = 3e-4
    alpha_lr: Optional[float] = 3e-4

    gamma: Optional[float] = 0.99
    tau: Optional[float] = 0.005
    reward_scale: Optional[float] = 1.0

    sil_weight: Optional[float] = 1.0