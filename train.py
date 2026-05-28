"""
Main training entry point for Repetition-RL.

Examples:
    python3 train.py run --gym openai --task HalfCheetah-v4 TD3
    python3 train.py run --gym openai --task HalfCheetah-v4 SAC
    python3 train.py run --gym openai --task HalfCheetah-v4 TD3SIL
    python3 train.py run --gym openai --task HalfCheetah-v4 SACSIL

    REPETITION_FRAMEWORK=IER SELECTION_STRATEGY=episode_reward \
    python3 train.py run --gym openai --task HalfCheetah-v4 ReTD3

    REPETITION_FRAMEWORK=IER SELECTION_STRATEGY=transition_reward \
    python3 train.py run --gym openai --task HalfCheetah-v4 ReTD3

    REPETITION_FRAMEWORK=SER SELECTION_STRATEGY=episode_reward \
    python3 train.py run --gym openai --task HalfCheetah-v4 ReTD3

    REPETITION_FRAMEWORK=SER SELECTION_STRATEGY=transition_reward \
    python3 train.py run --gym openai --task HalfCheetah-v4 ReTD3

    REPETITION_FRAMEWORK=SER SELECTION_STRATEGY=mixed \
    python3 train.py run --gym openai --task HalfCheetah-v4 ReTD3

    python3 train.py run --config configs/ser/mser_retd3.yaml
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

    run_parser.add_argument("algorithm", nargs="?", default=None)
    run_parser.add_argument("--config", type=str, default=None)

    run_parser.add_argument("--gym", type=str, default="openai")
    run_parser.add_argument("--task", type=str, default="Pendulum-v1")
    run_parser.add_argument("--domain", type=str, default="")
    run_parser.add_argument("--seeds", type=int, nargs="+", default=[10])

    return parser.parse_args()


def load_yaml_config(config_path):
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def get_repetition_config():
    repetition_framework = os.getenv("REPETITION_FRAMEWORK", "NONE").upper()
    selection_strategy = os.getenv("SELECTION_STRATEGY", "episode_reward").lower()

    return repetition_framework, selection_strategy


def get_algorithm_config(algorithm):
    config_map = {
        "TD3": TD3Config,
        "SAC": SACConfig,
        "TD3SIL": TD3SILConfig,
        "SACSIL": SACSILConfig,
        "ReTD3": ReTD3Config,
        "ReSAC": ReSACConfig,
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

    if alg_config.algorithm not in ["ReTD3", "ReSAC"]:
        return standard_policy_loop, "standard"

    if repetition_framework == "IER":
        if selection_strategy == "episode_reward":
            import train_loops.ier.episode_reward_loop as loop

            return loop, "ier_episode_reward"

        if selection_strategy == "transition_reward":
            import train_loops.ier.transition_reward_loop as loop

            return loop, "ier_transition_reward"

        raise ValueError(f"Unknown IER selection strategy: {selection_strategy}")

    if repetition_framework == "SER":
        if selection_strategy == "episode_reward":
            import train_loops.ser.eser.episode_reward_loop as loop

            return loop, "eser_episode_reward"

        if selection_strategy == "transition_reward":
            import train_loops.ser.xser.transition_reward_loop as loop

            return loop, "xser_transition_reward"

        if selection_strategy == "mixed":
            import train_loops.ser.mser.mixed_reward_loop as loop

            return loop, "mser_mixed"

        if selection_strategy == "simple_mixed":
            import train_loops.ser.mser.simple_mixed_reward_loop as loop

            return loop, "mser_simple_mixed"

        raise ValueError(f"Unknown SER selection strategy: {selection_strategy}")

    raise ValueError(
        f"{alg_config.algorithm} is a repetition algorithm. "
        "Set REPETITION_FRAMEWORK=IER or REPETITION_FRAMEWORK=SER."
    )


def build_configs_from_yaml(config_path):
    yaml_config = load_yaml_config(config_path)

    experiment_cfg = yaml_config.get("experiment", {})
    environment_cfg = yaml_config.get("environment", {})
    algorithm_cfg = yaml_config.get("algorithm", {})
    training_cfg = yaml_config.get("training", {})

    framework = experiment_cfg.get("framework", "NONE")
    selection_strategy = experiment_cfg.get(
        "selection_strategy",
        "episode_reward",
    )

    os.environ["REPETITION_FRAMEWORK"] = str(framework).upper()
    os.environ["SELECTION_STRATEGY"] = str(selection_strategy).lower()

    env_config = GymEnvironmentConfig(
        task=environment_cfg.get("task", "Pendulum-v1"),
        gym=environment_cfg.get("gym", "openai"),
        domain=environment_cfg.get("domain", ""),
    )

    training_config = TrainingConfig(
        seeds=training_cfg.get("seeds", [10]),
    )

    algorithm_name = algorithm_cfg.get("name")

    if algorithm_name is None:
        raise ValueError("Algorithm name missing in YAML config.")

    alg_config = get_algorithm_config(algorithm_name)

    return env_config, training_config, alg_config


def build_configs_from_cli(args):
    if args.algorithm is None:
        raise ValueError("Algorithm is required when not using --config.")

    env_config = GymEnvironmentConfig(
        task=args.task,
        gym=args.gym,
        domain=args.domain,
    )

    training_config = TrainingConfig(seeds=args.seeds)
    alg_config = get_algorithm_config(args.algorithm)

    return env_config, training_config, alg_config


def log_config(title, config):
    logging.info(
        "\n---------------------------------------------------\n"
        f"{title}\n"
        "---------------------------------------------------"
    )
    logging.info("\n%s", yaml.dump(dict(config), default_flow_style=False))


def main():
    args = parse_args()

    if args.config is not None:
        env_config, training_config, alg_config = build_configs_from_yaml(args.config)
    else:
        env_config, training_config, alg_config = build_configs_from_cli(args)

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
