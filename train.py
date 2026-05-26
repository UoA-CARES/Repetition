"""
Main training entry point for Repetition-RL.

Supported algorithms:
    TD3
    SAC
    ReTD3
    ReSAC
    TD3SIL
    SACSIL

Supported repetition settings for now:
    REPETITION_FRAMEWORK=NONE
    REPETITION_FRAMEWORK=IER

    SELECTION_STRATEGY=episode_reward
"""

import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import torch
import yaml

import train_loops.base.policy_loop as standard_policy_loop
import train_loops.ier.episode_reward_loop as ier_episode_reward_loop

from memory import EpisodicMemory, ReplayBuffer
from networks.network_factory import NetworkFactory
from utils import helpers as hlp
from utils.configurations import (
    GymEnvironmentConfig,
    ReSACConfig,
    ReTD3Config,
    SACConfig,
    SACSILConfig,
    TD3Config,
    TD3SILConfig,
    TrainingConfig,
)

logging.basicConfig(level=logging.INFO)


def get_repetition_config():
    repetition_framework = os.getenv("REPETITION_FRAMEWORK", "NONE").upper()
    selection_strategy = os.getenv("SELECTION_STRATEGY", "episode_reward").lower()

    return repetition_framework, selection_strategy


def get_algorithm_config():
    algorithm = os.getenv("ALGORITHM", "TD3")

    config_map = {
        "TD3": TD3Config,
        "SAC": SACConfig,
        "ReTD3": ReTD3Config,
        "ReSAC": ReSACConfig,
        "TD3SIL": TD3SILConfig,
        "SACSIL": SACSILConfig,
    }

    if algorithm not in config_map:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    return config_map[algorithm]()


def get_memory(alg_config):
    if alg_config.algorithm in ["ReTD3", "ReSAC", "TD3SIL", "SACSIL"]:
        return EpisodicMemory(
            replay_capacity=alg_config.buffer_size,
            vem_capacity=int(os.getenv("VEM_CAPACITY", 500)),
        )

    return ReplayBuffer(max_capacity=alg_config.buffer_size)


def get_training_loop(alg_config):
    repetition_framework, selection_strategy = get_repetition_config()

    repetition_algorithms = ["ReTD3", "ReSAC", "TD3SIL", "SACSIL"]

    if alg_config.algorithm in repetition_algorithms:
        if repetition_framework == "IER" and selection_strategy == "episode_reward":
            return ier_episode_reward_loop, "ier_episode_reward"

        raise ValueError(
            f"{alg_config.algorithm} requires REPETITION_FRAMEWORK=IER "
            "and SELECTION_STRATEGY=episode_reward for now."
        )

    return standard_policy_loop, "standard"


def log_config(title, config):
    logging.info(
        "\n---------------------------------------------------\n"
        f"{title}\n"
        "---------------------------------------------------"
    )
    logging.info(f"\n{yaml.dump(dict(config), default_flow_style=False)}")


def main():
    env_config = GymEnvironmentConfig(
        task=os.getenv("TASK", "Pendulum-v1"),
        gym=os.getenv("GYM", "gymnasium"),
        domain=os.getenv("DOMAIN", ""),
    )

    training_config = TrainingConfig()
    alg_config = get_algorithm_config()

    training_loop, repetition_name = get_training_loop(alg_config)

    iterations_folder = (
        f"{alg_config.algorithm}-{env_config.task}-"
        f"{datetime.now().strftime('%y_%m_%d_%H-%M-%S')}-"
        f"{repetition_name}"
    )

    glob_log_dir = f"{Path.home()}/repetition_rl_logs/{iterations_folder}"

    log_config("ENVIRONMENT CONFIG", env_config)
    log_config("ALGORITHM CONFIG", alg_config)
    log_config("TRAINING CONFIG", training_config)

    repetition_framework, selection_strategy = get_repetition_config()

    logging.info(
        "\n---------------------------------------------------\n"
        "REPETITION CONFIG\n"
        "---------------------------------------------------"
    )
    logging.info("REPETITION_FRAMEWORK: %s", repetition_framework)
    logging.info("SELECTION_STRATEGY: %s", selection_strategy)
    logging.info("Selected training loop: %s", repetition_name)

    logging.info(
        "Device: %s",
        torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )

    input("Double check your experiment configurations. Press ENTER to continue.")

    if not torch.cuda.is_available():
        no_gpu_answer = input(
            "No CUDA detected. Training will be slow. Continue? [y/n] "
        )

        if no_gpu_answer not in ["y", "Y"]:
            logging.info("Terminating experiment. CUDA is not available.")
            sys.exit()

    raise NotImplementedError(
        "Environment creation is the next step. Add an independent "
        "environment factory before running full experiments."
    )

    for training_iteration, seed in enumerate(training_config.seeds):
        logging.info(
            "Training iteration %s/%s with seed %s",
            training_iteration + 1,
            len(training_config.seeds),
            seed,
        )

        hlp.set_seed(seed)
        env.set_seed(seed)

        agent = NetworkFactory.create(
            algorithm_name=alg_config.algorithm,
            observation_size=env.observation_space,
            action_num=env.action_num,
            config=alg_config,
        )

        memory = get_memory(alg_config)

        record = None

        training_loop.policy_based_train(
            env,
            agent,
            memory,
            record,
            training_config,
            alg_config,
        )


if __name__ == "__main__":
    main()
