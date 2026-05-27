"""
Simple MSER training loop.

This mixed repetition strategy combines:
- Episode-reward repetition memory
- Transition-reward repetition memory

The first half of repetitions replay episode-reward memories,
while the second half replay transition-reward memories.
"""

import logging
import time

from utils import helpers as hlp
from utils.configurations import (
    AlgorithmConfig,
    TrainingConfig,
)


def evaluate_policy_network(
    env,
    agent,
    config: TrainingConfig,
    record=None,
    total_steps=0,
):
    number_eval_episodes = int(config.number_eval_episodes)

    state = env.reset()

    for eval_episode_counter in range(number_eval_episodes):

        episode_timesteps = 0
        episode_reward = 0
        episode_num = 0

        done = False
        truncated = False

        while not done and not truncated:

            episode_timesteps += 1

            action = agent.select_action_from_policy(
                state,
                evaluation=True,
            )

            action_env = hlp.denormalize(
                action,
                env.max_action_value,
                env.min_action_value,
            )

            state, reward, done, truncated = env.step(action_env)

            episode_reward += reward

            if done or truncated:

                if record is not None:
                    record.log_eval(
                        total_steps=total_steps + 1,
                        episode=eval_episode_counter + 1,
                        episode_reward=episode_reward,
                        display=True,
                    )

                state = env.reset()

                episode_reward = 0
                episode_timesteps = 0
                episode_num += 1


def policy_based_train(
    env,
    agent,
    memory,
    record,
    train_config: TrainingConfig,
    alg_config: AlgorithmConfig,
):
    start_time = time.time()

    explore = False
    crucial_steps = False

    max_steps_training = alg_config.max_steps_training
    max_steps_exploration = alg_config.max_steps_exploration

    number_steps_per_evaluation = (
        train_config.number_steps_per_evaluation
    )

    number_steps_per_train_policy = (
        alg_config.number_steps_per_train_policy
    )

    number_of_crusial_episodes = 0

    intrinsic_on = (
        bool(alg_config.intrinsic_on)
        if hasattr(alg_config, "intrinsic_on")
        else False
    )

    min_noise = (
        alg_config.min_noise
        if hasattr(alg_config, "min_noise")
        else 0
    )

    noise_decay = (
        alg_config.noise_decay
        if hasattr(alg_config, "noise_decay")
        else 1.0
    )

    noise_scale = (
        alg_config.noise_scale
        if hasattr(alg_config, "noise_scale")
        else 0.1
    )

    logging.info(
        "[CONFIG] Training=%s | Exploration=%s | Evaluation=%s",
        max_steps_training,
        max_steps_exploration,
        number_steps_per_evaluation,
    )

    batch_size = alg_config.batch_size
    G = alg_config.G

    episode_timesteps = 0
    episode_reward = 0
    episode_num = 0

    crucial_episode_num = 0
    crucial_total_reward = 0

    crucial_actions = []
    crucial_states = []

    save_episode = False

    RF = 10000
    RN = 5

    evaluate = False

    state = env.reset()

    episode_start = time.time()

    for total_step_counter in range(int(max_steps_training)):

        episode_timesteps += 1

        if total_step_counter % 1000 == 0:
            logging.info(
                "[TRAINING] Step=%s | Noise=%.4f",
                total_step_counter,
                noise_scale,
            )

        # =========================================================
        # Exploration
        # =========================================================

        if total_step_counter < max_steps_exploration or explore:

            logging.info(
                "[EXPLORATION] Step %s/%s",
                total_step_counter + 1,
                max_steps_exploration,
            )

            action_env = env.sample_action()

            explore = False

            action = hlp.normalize(
                action_env,
                env.max_action_value,
                env.min_action_value,
            )

        # =========================================================
        # Repetition
        # =========================================================

        elif crucial_steps and number_of_crusial_episodes > 0:

            if memory.long_term_memory.is_empty():

                crucial_steps = False

                print(
                    "[WARNING] Long-term memory is empty."
                )

                continue

            if episode_timesteps == 1:

                # First half = episode reward memory
                if number_of_crusial_episodes > RN / 2:

                    (
                        crucial_actions,
                        crucial_states,
                        crucial_episode_num,
                        episode_steps,
                        crucial_rewards,
                        crucial_total_reward,
                    ) = memory.long_term_memory.get_crucial_path(1)

                    print(
                        "[REPLAY START] "
                        "Mode=EpisodeReward"
                    )

                # Second half = transition reward memory
                else:

                    (
                        crucial_actions,
                        crucial_states,
                        crucial_episode_num,
                        episode_steps,
                        crucial_rewards,
                        crucial_total_reward,
                    ) = memory.long_term_memory_total.get_crucial_path(1)

                    print(
                        "[REPLAY START] "
                        "Mode=TransitionReward"
                    )

            action = crucial_actions[episode_timesteps - 1]

            action_env = hlp.denormalize(
                action,
                env.max_action_value,
                env.min_action_value,
            )

            if episode_timesteps >= len(crucial_actions):

                crucial_steps = False

                print(
                    "[REPETITION END] "
                    f"RemainingRepeats="
                    f"{number_of_crusial_episodes}"
                )

        # =========================================================
        # Policy action
        # =========================================================

        else:

            noise_scale *= noise_decay

            noise_scale = max(
                min_noise,
                noise_scale,
            )

            action = agent.select_action_from_policy(
                state,
                noise_scale=noise_scale,
            )

            action_env = hlp.denormalize(
                action,
                env.max_action_value,
                env.min_action_value,
            )

        # =========================================================
        # Environment step
        # =========================================================

        next_state, reward_extrinsic, done, truncated = (
            env.step(action_env)
        )

        intrinsic_reward = 0

        if (
            intrinsic_on
            and total_step_counter > max_steps_exploration
        ):
            intrinsic_reward = agent.get_intrinsic_reward(
                state,
                action,
                next_state,
            )

        total_reward = (
            reward_extrinsic + intrinsic_reward
        )

        # =========================================================
        # Short-term memory
        # =========================================================

        memory.short_term_memory.add(
            state,
            action,
            total_reward,
            next_state,
            done,
            episode_num,
            episode_timesteps,
        )

        state = next_state

        episode_reward += reward_extrinsic

        # =========================================================
        # Transition reward memory
        # =========================================================

        if (
            total_step_counter > batch_size
            and total_reward > 0
        ):

            if (
                not memory.long_term_memory_total.is_full()
            ) or (
                memory.long_term_memory_total.is_full()
                and total_reward >
                memory.long_term_memory_total.get_min_reward()
                and episode_timesteps > 2
            ):

                (
                    states,
                    actions,
                    rewards,
                    next_states,
                    dones,
                    episode_nums,
                    episode_steps,
                ) = memory.short_term_memory.sample_complete_episode(
                    episode_num,
                    episode_timesteps,
                )

                memory.long_term_memory_total.add(
                    [
                        episode_num,
                        total_reward,
                        states,
                        actions,
                        rewards,
                        next_states,
                        dones,
                        episode_nums,
                        episode_steps,
                    ]
                )

                print(
                    "[MEMORY ADD] "
                    "Buffer=TransitionReward | "
                    f"Episode={episode_num} | "
                    f"TransitionReward={total_reward:.2f}"
                )

                save_episode = True

        # =========================================================
        # Policy training
        # =========================================================

        if (
            total_step_counter >= max_steps_exploration
            and total_step_counter %
            number_steps_per_train_policy == 0
        ):

            for _ in range(G):

                agent.train_policy(
                    memory,
                    batch_size,
                )

        # =========================================================
        # Evaluation trigger
        # =========================================================

        if (
            total_step_counter + 1
        ) % number_steps_per_evaluation == 0:

            evaluate = True

        # =========================================================
        # Repetition trigger
        # =========================================================

        if (
            (total_step_counter + 1) % RF == 0
            and episode_reward > 0
        ):

            number_of_crusial_episodes = RN + 1

            print(
                "[REPETITION TRIGGER] "
                f"Repeats={RN}"
            )

        # =========================================================
        # Episode end
        # =========================================================

        if done or truncated:

            print(
                "[EPISODE END] "
                f"Steps={episode_timesteps} | "
                f"ReplayLength={len(crucial_actions)} | "
                f"RemainingRepeats="
                f"{number_of_crusial_episodes}"
            )

            episode_time = (
                time.time() - episode_start
            )

            record.log_train(
                total_steps=total_step_counter + 1,
                episode=episode_num + 1,
                episode_steps=episode_timesteps,
                episode_reward=episode_reward,
                episode_time=episode_time,
                display=True,
            )

            # =============================================
            # Repetition completion
            # =============================================

            if number_of_crusial_episodes == 1:

                crucial_steps = False

                number_of_crusial_episodes -= 1

                print(
                    "[REPETITION COMPLETE] "
                    f"Steps={episode_timesteps} | "
                    f"TransitionReward="
                    f"{total_reward:.2f} | "
                    f"EpisodeReward="
                    f"{episode_reward:.2f} | "
                    f"StoredReward="
                    f"{crucial_total_reward:.2f}"
                )

            elif number_of_crusial_episodes > 1:

                number_of_crusial_episodes -= 1

                crucial_steps = True

                print(
                    "[REPLAY EPISODE] "
                    f"TransitionReward="
                    f"{total_reward:.2f} | "
                    f"EpisodeReward="
                    f"{episode_reward:.2f}"
                )

            # =============================================
            # Episode reward memory
            # =============================================

            elif (
                not memory.long_term_memory.is_full()
                and not crucial_steps
                and episode_reward > 0
            ) or (
                not crucial_steps
                and memory.long_term_memory.is_full()
                and episode_reward >
                memory.long_term_memory.get_min_reward()
                and episode_timesteps > 2
            ):

                (
                    states,
                    actions,
                    rewards,
                    next_states,
                    dones,
                    episode_nums,
                    episode_steps,
                ) = memory.short_term_memory.sample_complete_episode(
                    episode_num,
                    episode_timesteps,
                )

                memory.long_term_memory.add(
                    [
                        episode_num,
                        episode_reward,
                        states,
                        actions,
                        rewards,
                        next_states,
                        dones,
                        episode_nums,
                        episode_steps,
                    ]
                )

                print(
                    "[MEMORY ADD] "
                    "Buffer=EpisodeReward | "
                    f"Episode={episode_num} | "
                    f"EpisodeReward="
                    f"{episode_reward:.2f}"
                )

            # =============================================
            # Evaluation
            # =============================================

            if evaluate:

                logging.info(
                    "[EVALUATION] "
                    "Starting evaluation "
                    "at step %s",
                    total_step_counter + 1,
                )

                evaluate_policy_network(
                    env,
                    agent,
                    train_config,
                    record=record,
                    total_steps=total_step_counter,
                )

                logging.info(
                    "[EVALUATION] Finished"
                )

                evaluate = False

            # =============================================
            # Reset episode
            # =============================================

            state = env.reset()

            episode_timesteps = 0
            episode_reward = 0

            episode_num += 1

            episode_start = time.time()

    elapsed_time = (
        time.time() - start_time
    )

    print(
        "[TRAINING COMPLETE] "
        f"TotalTime="
        f"{time.strftime('%H:%M:%S', time.gmtime(elapsed_time))}"
    )
