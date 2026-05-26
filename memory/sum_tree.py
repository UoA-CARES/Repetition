"""
SumTree data structure used for prioritized experience replay.
"""

import math
import random

import numpy as np


class SumTree:
    """
    Binary sum tree for efficient prioritized sampling.
    """

    def __init__(self, max_size: int):
        self.levels = [np.zeros(1)]

        level_size = 1

        while level_size < max_size:
            level_size *= 2
            self.levels.append(np.zeros(level_size))

    def sample_value(self, query_value: int = None) -> int:
        """
        Sample a single index from the tree.
        """

        query_value = random.random() if query_value is None else query_value
        query_value *= self.levels[0][0]

        return self._retrieve([query_value])[0]

    def sample_simple(self, batch_size: int) -> list[int]:
        """
        Uniform priority sampling.
        """

        values = np.random.uniform(
            0,
            self.levels[0][0],
            size=batch_size,
        )

        return self._retrieve(values)

    def sample_stratified(self, batch_size: int) -> list[int]:
        """
        Stratified priority sampling.
        """

        bounds = np.linspace(0.0, 1.0, batch_size + 1)

        segments = [
            (bounds[i], bounds[i + 1])
            for i in range(batch_size)
        ]

        query_values = [
            random.uniform(segment[0], segment[1]) * self.levels[0][0]
            for segment in segments
        ]

        return self._retrieve(query_values)

    def _retrieve(self, values: np.ndarray) -> list[int]:
        """
        Retrieve indices corresponding to cumulative priorities.
        """

        ind = np.zeros(len(values), dtype=int)

        for nodes in self.levels[1:]:
            ind *= 2

            left_sum = nodes[ind]

            is_greater = np.greater(values, left_sum)

            ind += is_greater

            values -= left_sum * is_greater

        return ind

    def set(self, ind: int, new_priority: float) -> None:
        """
        Update a single priority value.
        """

        priority_diff = (
            new_priority - self.levels[-1][ind]
        )

        for nodes in self.levels[::-1]:
            np.add.at(nodes, ind, priority_diff)
            ind //= 2

    def batch_set(
        self,
        ind: list[int],
        new_priority: list[float],
    ) -> None:
        """
        Batch update priorities.
        """

        ind, unique_ind = np.unique(
            ind,
            return_index=True,
        )

        priority_diff = (
            new_priority[unique_ind]
            - self.levels[-1][ind]
        )

        for nodes in self.levels[::-1]:
            np.add.at(nodes, ind, priority_diff)
            ind //= 2

    def batch_set_v2(
        self,
        ind: list[int],
        new_priority: list[float],
    ) -> None:

        max_ind_value = ind[-1]

        if len(ind) % 2 == 0:

            loop_counter = len(self.levels[::-1])

            for i in range(loop_counter):

                if i == 0:

                    self.levels[::-1][i][:len(new_priority)] = (
                        new_priority
                    )

                    max_ind_value //= 2

                else:

                    check_cond_1 = max_ind_value + 1

                    if i == 1:
                        len_priorities = len(new_priority)
                    else:
                        len_priorities = len(
                            self.levels[::-1][i - 1][0:dummy]
                        )

                    if (
                        math.ceil(len_priorities / 2)
                        == check_cond_1
                    ):

                        if i == 1:
                            self.levels[::-1][i][:check_cond_1] = (
                                new_priority[0:len_priorities:2]
                            )
                        else:
                            self.levels[::-1][i][:check_cond_1] = (
                                self.levels[::-1][i - 1][0:dummy][
                                    0:len_priorities:2
                                ]
                            )

                    else:

                        if i == 1:
                            self.levels[::-1][i][:check_cond_1 - 1] = (
                                new_priority[0:len_priorities:2]
                            )
                        else:
                            self.levels[::-1][i][:check_cond_1 - 1] = (
                                self.levels[::-1][i - 1][0:dummy][
                                    0:len_priorities:2
                                ]
                            )

                    if (
                        math.floor(len_priorities / 2)
                        == check_cond_1
                    ):

                        if i == 1:
                            self.levels[::-1][i][:check_cond_1] += (
                                new_priority[1:len_priorities:2]
                            )
                        else:
                            self.levels[::-1][i][:check_cond_1] += (
                                self.levels[::-1][i - 1][0:dummy][
                                    1:len_priorities:2
                                ]
                            )

                    else:

                        if i == 1:
                            self.levels[::-1][i][:check_cond_1 - 1] += (
                                new_priority[1:len_priorities:2]
                            )
                        else:
                            self.levels[::-1][i][:check_cond_1 - 1] += (
                                self.levels[::-1][i - 1][0:dummy][
                                    1:len_priorities:2
                                ]
                            )

                    dummy = len_priorities // 2

                    if dummy == 1 or dummy == 0:
                        dummy = 2

                    max_ind_value //= 2