"""
Valuable Episode Memory (VEM).

This module stores valuable complete episodes for repetition-based RL.

The stored episode format follows the original implementation:

    (
        episode_num,
        episode_reward,
        states,
        actions,
        rewards,
        next_states,
        dones,
        episode_nums,
        episode_steps,
    )
"""
Memory manager for repetition-based RL.

This class groups the memory components needed by repetition methods:
standard replay memory, short-term episode memory, and Valuable Episode Memory.
"""

from memory.short_term_memory import ShortTermMemory
from memory.valuable_episode_memory import (
    ValuableEpisodeMemory,
    ValuableEpisodeMemoryTotal,
)


class MemoryManager:
    """
    Container for all memory modules used by IER and SER methods.

    Attributes:
        short_term_memory:
            Temporarily records transitions with episode ids and steps.

        valuable_episode_memory:
            Stores valuable episodes selected using episode reward.

        valuable_episode_memory_total:
            Stores valuable episodes selected using total reward.
    """

    def __init__(self, max_capacity: int = int(1e6), vem_capacity: int = 500, **kwargs):
        self.short_term_memory = ShortTermMemory(
            max_capacity=max_capacity,
            **kwargs,
        )

        self.valuable_episode_memory = ValuableEpisodeMemory(
            max_capacity=vem_capacity,
            **kwargs,
        )

        self.valuable_episode_memory_total = ValuableEpisodeMemoryTotal(
            max_capacity=vem_capacity,
            **kwargs,
        )

        # Backward-compatible aliases for old training loops.
        self.long_term_memory = self.valuable_episode_memory
        self.long_term_memory_total = self.valuable_episode_memory_total
Both classes below keep the same behaviour as the original LongMemoryBuffer
and LongMemoryBufferTotal, but use clearer naming and cleaner comments.
"""

import random
from collections import deque

import numpy as np


class ValuableEpisodeMemory:
    """
    Stores high-value episodes according to an episode-level reward score.

    This replaces the original LongMemoryBuffer class.
    """

    def __init__(self, max_capacity: int = int(1e2), **priority_params):
        self.priority_params = priority_params
        self.max_capacity = max_capacity

        self.memory_buffers = deque([], maxlen=self.max_capacity)

        self.min_high_reward = float("inf")
        self.max_reward = -float("inf")
        self.min_index = -1

    def add(self, experience) -> None:
        """
        Add a complete episode to memory.

        If the memory is full, the new episode replaces the current lowest
        reward episode only when its reward is higher.
        """

        episode_reward = experience[1]

        if episode_reward > self.max_reward:
            self.max_reward = episode_reward

        if self.is_full():
            if episode_reward > self.min_high_reward:
                self.memory_buffers[self.min_index] = experience
                self.update_min_reward()
        else:
            self.memory_buffers.append(experience)

            if episode_reward < self.min_high_reward:
                self.min_high_reward = episode_reward
                self.min_index = len(self.memory_buffers) - 1

    def update_min_reward(self) -> None:
        """Update the current minimum reward and its index."""

        if self.memory_buffers:
            rewards = [entry[1] for entry in self.memory_buffers]
            self.min_index = int(np.argmin(rewards))
            self.min_high_reward = rewards[self.min_index]
        else:
            self.min_high_reward = float("inf")
            self.min_index = -1

    def is_full(self) -> bool:
        """Return True if the memory has reached its capacity."""

        return len(self.memory_buffers) >= self.max_capacity

    def is_empty(self) -> bool:
        """Return True if the memory is empty."""

        return len(self.memory_buffers) == 0

    def get_length(self) -> int:
        """Return the number of stored episodes."""

        return len(self.memory_buffers)

    def sample_uniform(self, batch_size: int) -> list:
        """
        Sample episodes uniformly with replacement.

        This keeps the original behaviour, where random indices are sampled
        independently rather than using random.sample without replacement.
        """

        if self.is_empty():
            raise ValueError("Valuable Episode Memory is empty.")

        selected_experiences = []
        buffer_length = len(self.memory_buffers)

        for _ in range(batch_size):
            index = random.randint(0, buffer_length - 1)
            selected_experiences.append(self.memory_buffers[index])

        return selected_experiences

    def sample_max_reward(self):
        """Return the episode with the highest stored reward."""

        for experience in self.memory_buffers:
            if experience[1] == self.max_reward:
                return experience

        return None

    def sample_single(self):
        """Sample one random episode from memory."""

        if self.is_empty():
            raise ValueError("Valuable Episode Memory is empty.")

        return random.choice(self.memory_buffers)

    def get_min_reward(self) -> float:
        """Return the lowest reward among currently stored valuable episodes."""

        return self.min_high_reward

    def get_max_crucial_path(self, number_of_crucial_episodes: int):
        """
        Return the action sequence from the highest-reward episode.

        The argument is kept for compatibility with the original code.
        """

        episode_batch = self.sample_max_reward()

        if episode_batch is None:
            raise ValueError("No episode with max reward found.")

        (
            episode_num,
            total_reward,
            states,
            actions,
            rewards,
            next_states,
            dones,
            episode_nums,
            episode_steps,
        ) = episode_batch

        return actions, states, episode_num, episode_steps, rewards, total_reward

    def get_crucial_path(self, number_of_crucial_episodes: int):
        """
        Return the action sequence from one randomly selected valuable episode.

        The argument is kept for compatibility with the original code.
        """

        episode_batch = self.sample_single()

        if episode_batch is None:
            raise ValueError("No episode found.")

        (
            episode_num,
            total_reward,
            states,
            actions,
            rewards,
            next_states,
            dones,
            episode_nums,
            episode_steps,
        ) = episode_batch

        return actions, states, episode_num, episode_steps, rewards, total_reward

    def sample_neighbour(self, episode_num: int, episode_steps: int):
        """
        Return a stored episode matching the provided episode id and step.

        This keeps the behaviour of the original sample_neighbour method.
        """

        for experience in self.memory_buffers:
            if experience[0] == episode_num and experience[7] == episode_steps:
                return experience

        return None


class ValuableEpisodeMemoryTotal(ValuableEpisodeMemory):
    """
    Stores high-value episodes using the total-reward variant.

    This replaces the original LongMemoryBufferTotal class. The behaviour is
    currently identical to ValuableEpisodeMemory, but the separate class name is
    preserved to keep total-reward experiments explicit and readable.
    """

    pass
