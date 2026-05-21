"""
Standard replay buffer for off-policy RL algorithms.

This buffer stores individual transitions and is used for policy/value updates
in algorithms such as TD3 and SAC.
"""

import random
from collections import deque


class ReplayBuffer:
    """A simple uniform replay buffer."""

    def __init__(self, max_capacity: int = int(1e6), **kwargs):
        self.max_capacity = max_capacity
        self.buffer = deque(maxlen=max_capacity)

    def add(self, state, action, reward, next_state, done) -> None:
        """Add a transition to the replay buffer."""
        self.buffer.append((state, action, reward, next_state, done))

    def sample_uniform(self, batch_size: int):
        """Sample a batch of transitions uniformly."""
        if batch_size > len(self.buffer):
            raise ValueError(
                f"Cannot sample batch_size={batch_size} from buffer of size {len(self.buffer)}."
            )

        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        return len(self.buffer)
