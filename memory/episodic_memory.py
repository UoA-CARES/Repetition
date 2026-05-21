from memory.short_term_memory import ShortTermMemory
from memory.valuable_episode_memory import (
    ValuableEpisodeMemory,
    ValuableEpisodeMemoryTotal,
)


class EpisodicMemory:
    """
    Complete memory system used for repetition-based RL.

    Components:
        - Short-term transition memory
        - Valuable Episode Memory (VEM)
        - Total-reward VEM
    """

    def __init__(
        self,
        replay_capacity: int = int(1e6),
        vem_capacity: int = 500,
        **kwargs,
    ):
        self.short_term_memory = ShortTermMemory(
            max_capacity=replay_capacity,
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

        # backward compatibility
        self.long_term_memory = self.valuable_episode_memory
        self.long_term_memory_total = (
            self.valuable_episode_memory_total
        )
