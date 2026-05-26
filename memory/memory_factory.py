"""
Factory for creating replay memories.
"""

from memory.replay_buffer import ReplayBuffer
from memory.prioritized_replay_buffer import (
    PrioritizedReplayBuffer,
)

from memory.short_term_memory import ShortTermMemory
from memory.valuable_episode_memory import (
    ValuableEpisodeMemory,
)

from memory.episodic_memory import EpisodicMemory


class ManageBuffers:
    """
    Container class for repetition-based memories.
    """

    def __init__(self):

        self.short_term_memory = ShortTermMemory()

        self.valuable_memory = ValuableEpisodeMemory()

        self.episodic_memory = EpisodicMemory()


class MemoryFactory:

    @staticmethod
    def create_memory(config):

        algorithm = config.algorithm

        # ==========================================
        # Standard replay-based algorithms
        # ==========================================

        if algorithm in [
            "TD3",
            "SAC",
            "TD3SIL",
            "SACSIL",
        ]:
            return ReplayBuffer()

        # ==========================================
        # Repetition-based algorithms
        # ==========================================

        if algorithm in [
            "ReTD3",
            "ReSAC",
        ]:
            return ManageBuffers()

        # ==========================================
        # Prioritized replay algorithms
        # ==========================================

        if algorithm in [
            "PERTD3",
            "PERSAC",
        ]:
            return PrioritizedReplayBuffer()

        raise ValueError(
            f"Unknown memory configuration for algorithm: {algorithm}"
        )