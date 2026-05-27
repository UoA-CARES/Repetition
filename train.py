"""
Main training entry point for Repetition-RL.
"""

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import torch
import yaml

import train_loops.base.policy_loop as standard_policy_loop
import train_loops.ier.episode_reward_loop as ier_episode_reward_loop

from environments.environment_factory import EnvironmentFactory
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
from utils.record import Record

logging.basicConfig(level=logging.INFO, force=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Train Repetition-RL agents.")

    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run")

    run_parser.add_argument("algorithm", type=str)
    run_parser.add_argument("--gym", type=str, default="openai")
    run_parser.add_argument("--task", type=str, default="Pendulum-v1")
    run_parser.add_argument("--domain", type=str, default="")
    run_parser.add_argument("--seeds", type=int, nargs="+", default=[10])

    return parser.parse_args()


def get_repetition_config():
    repetition_framework = os.getenv("REPETITION_FRAMEWORK", "NONE").upper()
    selection_strategy = os.getenv("SELECTION_STRATEGY", "episode_reward").lower()

    return repetition_framework, selection_strategy


def get_algorithm_config(algorithm):
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
    if alg_config.algorithm in ["ReTD3", "ReSAC"]:
        return EpisodicMemory(
            replay_capacity=alg_config.buffer_size,
            vem_capacity=int(os.getenv("VEM_CAPACITY", 500)),
        )

    return ReplayBuffer(max_capacity=alg_config.buffer_size)


def get_training_loop(alg_config):
    repetition_framework, selection_strategy = get_repetition_config()

    if alg_config.algorithm in ["ReTD3", "ReSAC"]:
        if repetition_framework == "IER" and selection_strategy == "episode_reward":
            return ier_episode_reward_loop, "ier_episode_reward"

        raise ValueError(
            f"{alg_config.algorithm} requires REPETITION_FRAMEWORK=IER "
            "and SELECTION_STRATEGY=episode_reward."
        )

    return standard_policy_loop, "standard"


def log_config(title, config):
    logging.info(
        "\n---------------------------------------------------\n"
        f"{title}\n"
        "---------------------------------------------------"
    )
    logging.info("\n%s", yaml.dump(dict(config), default_flow_style=False))


def main():
    args = parse_args()

    env_config = GymEnvironmentConfig(
        task=args.task,
        gym=args.gym,
        domain=args.domain,
    )

    training_config = TrainingConfig(seeds=args.seeds)
    alg_config = get_algorithm_config(args.algorithm)

    training_loop, loop_name = get_training_loop(alg_config)

    iterations_folder = (
        f"{alg_config.algorithm}-{env_config.task}-"
        f"{datetime.now().strftime('%y_%m_%d_%H-%M-%S')}-"
        f"{loop_name}"
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
    logging.info("Selected training loop: %s", loop_name)
    logging.info("Log directory: %s", glob_log_dir)
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

    env_factory = EnvironmentFactory()

    for training_iteration, seed in enumerate(training_config.seeds):
        logging.info(
            "Training iteration %s/%s with seed %s",
            training_iteration + 1,
            len(training_config.seeds),
            seed,
        )

        env = env_factory.create_environment(env_config)

        hlp.set_seed(seed)
        env.set_seed(seed)

        agent = NetworkFactory.create(
            algorithm_name=alg_config.algorithm,
            observation_size=env.observation_space,
            action_num=env.action_num,
            config=alg_config,
        )

        memory = get_memory(alg_config)

        record = Record(
            glob_log_dir=glob_log_dir,
            log_dir=f"{seed}",
            algorithm=alg_config.algorithm,
            task=env_config.task,
            network=agent,
            plot_frequency=training_config.plot_frequency,
            checkpoint_frequency=training_config.checkpoint_frequency,
        )

        record.save_config(env_config, "env_config")
        record.save_config(training_config, "train_config")
        record.save_config(alg_config, "alg_config")

        training_loop.policy_based_train(
            env,
            agent,
            memory,
            record,
            training_config,
            alg_config,
        )

        record.save()


if __name__ == "__main__":
    main()