"""
Main training entry point for Repetition-RL.

This script trains RL agents across standard, IER, and SER repetition settings.

Environment variables:
    REPETITION_FRAMEWORK:
        NONE
        IER
        SER

    SELECTION_STRATEGY:
        episode_reward
        transition_reward
        mixed

    SER_VARIANT:
        standard
        delayed
        adaptive
        near_neighbor
        surprise

    ALGORITHM:
        TD3
        SAC

Examples:
    REPETITION_FRAMEWORK=IER SELECTION_STRATEGY=episode_reward ALGORITHM=TD3 python train.py

    REPETITION_FRAMEWORK=SER SELECTION_STRATEGY=episode_reward SER_VARIANT=standard ALGORITHM=SAC python train.py

    REPETITION_FRAMEWORK=SER SELECTION_STRATEGY=transition_reward SER_VARIANT=near_neighbor ALGORITHM=TD3 python train.py
"""

import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import torch
import yaml

import train_loops.policy_loop as ppe
import train_loops.value_loop as vbe
import train_loops.episodic_policy_loop as epbe

# IER loops
import train_loops.policy_loop_instant_repeat_episodic as ier_episode_reward_loop

# SER loops
import train_loops.policy_loop_repeat_episodic_di as ser_episode_reward_loop

from envrionments.environment_factory import EnvironmentFactory
from util.configurations import GymEnvironmentConfig

from cares_reinforcement_learning.memory.memory_factory import MemoryFactory
from cares_reinforcement_learning.util import NetworkFactory, Record, RLParser
from cares_reinforcement_learning.util import helpers as hlp


logging.basicConfig(level=logging.INFO)


def get_repetition_config():
    """Read repetition settings from environment variables."""

    repetition_framework = os.getenv("REPETITION_FRAMEWORK", "NONE").upper()
    selection_strategy = os.getenv("SELECTION_STRATEGY", "episode_reward").lower()
    ser_variant = os.getenv("SER_VARIANT", "standard").lower()

    return repetition_framework, selection_strategy, ser_variant


def get_repetition_loop():
    """
    Select the correct training loop based on repetition environment variables.

    Current supported working loops:
        IER + episode_reward
        SER + episode_reward

    Other variants can be added gradually as their loop files are migrated.
    """

    repetition_framework, selection_strategy, ser_variant = get_repetition_config()

    if repetition_framework == "NONE":
        return None, "standard"

    if repetition_framework == "IER":
        if selection_strategy == "episode_reward":
            return ier_episode_reward_loop, "ier_episode_reward"

        raise ValueError(
            f"Unsupported IER selection strategy: {selection_strategy}. "
            "Currently supported: episode_reward."
        )

    if repetition_framework == "SER":
        if selection_strategy == "episode_reward" and ser_variant == "standard":
            return ser_episode_reward_loop, "eser"

        raise ValueError(
            f"Unsupported SER setting: selection_strategy={selection_strategy}, "
            f"ser_variant={ser_variant}. Currently supported: "
            "SELECTION_STRATEGY=episode_reward and SER_VARIANT=standard."
        )

    raise ValueError(
        f"Unsupported REPETITION_FRAMEWORK={repetition_framework}. "
        "Use NONE, IER, or SER."
    )


def log_config(title, config):
    """Pretty-print a configuration block."""

    logging.info(
        "\n---------------------------------------------------\n"
        f"{title}\n"
        "---------------------------------------------------"
    )
    logging.info(f"\n{yaml.dump(dict(config), default_flow_style=False)}")


def main():
    """Parse configuration, create environment/agent/memory, and run training."""

    parser = RLParser(GymEnvironmentConfig)

    configurations = parser.parse_args()
    env_config = configurations["env_config"]
    training_config = configurations["training_config"]
    alg_config = configurations["algorithm_config"]

    repetition_loop, repetition_name = get_repetition_loop()

    env_factory = EnvironmentFactory()
    network_factory = NetworkFactory()
    memory_factory = MemoryFactory()

    iterations_folder = (
        f"{alg_config.algorithm}-{env_config.task}-"
        f"{datetime.now().strftime('%y_%m_%d_%H-%M-%S')}-"
        f"{repetition_name}"
    )
    glob_log_dir = f"{Path.home()}/cares_rl_logs/{iterations_folder}"

    log_config("ENVIRONMENT CONFIG", env_config)
    log_config("ALGORITHM CONFIG", alg_config)
    log_config("TRAINING CONFIG", training_config)

    logging.info(
        "\n---------------------------------------------------\n"
        "REPETITION CONFIG\n"
        "---------------------------------------------------"
    )
    repetition_framework, selection_strategy, ser_variant = get_repetition_config()
    logging.info(f"REPETITION_FRAMEWORK: {repetition_framework}")
    logging.info(f"SELECTION_STRATEGY: {selection_strategy}")
    logging.info(f"SER_VARIANT: {ser_variant}")
    logging.info(f"Selected training loop: {repetition_name}")

    logging.info(
        f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}"
    )

    input("Double check your experiment configurations. Press ENTER to continue.")

    if not torch.cuda.is_available():
        no_gpu_answer = input(
            "No CUDA detected. Training will be slow. Continue? [y/n] "
        )

        if no_gpu_answer not in ["y", "Y"]:
            logging.info("Terminating experiment. CUDA is not available.")
            sys.exit()

    for training_iteration, seed in enumerate(training_config.seeds):
        logging.info(
            f"Training iteration {training_iteration + 1}/"
            f"{len(training_config.seeds)} with seed: {seed}"
        )

        env = env_factory.create_environment(env_config)

        # Keep seed setup here for seed consistency.
        hlp.set_seed(seed)
        env.set_seed(seed)

        logging.info(f"Algorithm: {alg_config.algorithm}")

        agent = network_factory.create_network(
            env.observation_space,
            env.action_num,
            alg_config,
        )

        if agent is None:
            raise ValueError(f"Unknown agent: {alg_config.algorithm}")

        memory = memory_factory.create_memory(alg_config)

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

        if agent.type == "policy":
            if alg_config.algorithm == "EpisodicTD3":
                epbe.policy_based_train(
                    env, agent, memory, record, training_config, alg_config
                )

            elif alg_config.algorithm in ["ReTD3", "RESAC"]:
                if repetition_loop is None:
                    raise ValueError(
                        "ReTD3/RESAC requires a repetition loop. "
                        "Set REPETITION_FRAMEWORK=IER or REPETITION_FRAMEWORK=SER."
                    )

                repetition_loop.policy_based_train(
                    env, agent, memory, record, training_config, alg_config
                )

            else:
                ppe.policy_based_train(
                    env, agent, memory, record, training_config, alg_config
                )

        elif agent.type == "value":
            vbe.value_based_train(
                env, agent, memory, record, training_config, alg_config
            )

        else:
            raise ValueError(f"Unknown agent type: {agent.type}")

        record.save()


if __name__ == "__main__":
    main()
