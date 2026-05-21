"""
Short-term episode memory.

This memory temporarily stores transitions from the current interaction
episodes so complete trajectories can later be reconstructed and stored
inside the Virtual Episode Memory (VEM).
"""

import random
from collections import deque


class ShortTermMemory:
    """Temporary episode-level storage."""

    def __init__(self, max_capacity: int = int(1e6), **kwargs):
        self.max_capacity = max_capacity
        self.buffer = deque(maxlen=max_capacity)

    def add(
        self,
        state,
        action,
        reward,
        next_state,
        done,
        episode_num,
        episode_step,
    ) -> None:
        """Store a transition with episode information."""

        experience = (
            state,
            action,
            reward,
            next_state,
            done,
            episode_num,
            episode_step,
        )

        self.buffer.append(experience)

    def sample_uniform(self, batch_size: int):
        """Uniformly sample transitions."""

        batch = random.sample(self.buffer, batch_size)

        (
            states,
            actions,
            rewards,
            next_states,
            dones,
            episode_nums,
            episode_steps,
        ) = zip(*batch)

        return (
            states,
            actions,
            rewards,
            next_states,
            dones,
            episode_nums,
            episode_steps,
        )

    def sample_complete_episode(
        self,
        target_episode_num: int,
        target_episode_step: int,
    ):
        """
        Retrieve a complete episode trajectory.
        """

        start_idx = None
        end_idx = None

        for i, experience in enumerate(self.buffer):
            episode_num = experience[-2]
            episode_step = experience[-1]

            if episode_num == target_episode_num:

                if episode_step == 1:
                    start_idx = i

                if episode_step == target_episode_step:
                    end_idx = i + 1
                    break

        if start_idx is None or end_idx is None:
            raise ValueError("No matching episode found.")

        batch = list(self.buffer)[start_idx:end_idx]

        (
            states,
            actions,
            rewards,
            next_states,
            dones,
            episode_nums,
            episode_steps,
        ) = zip(*batch)

        return (
            states,
            actions,
            rewards,
            next_states,
            dones,
            episode_nums,
            episode_steps,
        )

    def sample_episode_segment(
        self,
        target_episode_num: int,
        target_episode_step: int,
        batch_size: int,
    ):
        """
        Retrieve a partial trajectory segment from an episode.
        """

        matching_index = None

        for i, experience in enumerate(self.buffer):
            episode_num = experience[-2]
            episode_step = experience[-1]

            if (
                episode_num == target_episode_num
                and episode_step == target_episode_step
            ):
                matching_index = i
                break

        if matching_index is None:
            raise ValueError("No matching transition found.")

        if matching_index < batch_size:
            start_idx = max(
                0,
                matching_index - target_episode_step + 1,
            )
        else:
            start_idx = max(
                0,
                matching_index - batch_size,
            )

        end_idx = matching_index + 1

        batch = list(self.buffer)[start_idx:end_idx]

        (
            states,
            actions,
            rewards,
            next_states,
            dones,
            episode_nums,
            episode_steps,
        ) = zip(*batch)

        return (
            states,
            actions,
            rewards,
            next_states,
            dones,
            episode_nums,
            episode_steps,
        )

    def __len__(self):
        return len(self.buffer)
