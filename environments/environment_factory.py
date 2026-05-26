"""
Environment factory for Repetition-RL.
"""

import logging

from environments.dm_control.dmcs_environment import DMCSEnvironment
from environments.mujoco.openai_environment import OpenAIEnvironment
from environments.wrappers.image_wrapper import ImageWrapper


class EnvironmentFactory:
    """Creates training environments."""

    def create_environment(self, config):
        logging.info("Training environment backend: %s", config.gym)

        if config.gym in ["gymnasium", "openai", "mujoco"]:
            env = OpenAIEnvironment(config)

        elif config.gym in ["dmcs", "dm_control"]:
            env = DMCSEnvironment(config)

        else:
            raise ValueError(f"Unknown environment backend: {config.gym}")

        if hasattr(config, "image_observation") and bool(config.image_observation):
            return ImageWrapper(env)

        return env