"""
Immediate Episode Repetition (IER), episode-reward selection.

This loop repeats newly discovered high-reward episodes immediately. An episode
is selected for repetition when its cumulative extrinsic reward is higher than
the best episode reward observed so far.
"""

import logging
import time

from utils import helpers as hlp
from utils.configurations import AlgorithmConfig, TrainingConfig


def evaluate_policy_network(
    env,
    agent,
    config: TrainingConfig,
    record=None,
    total_steps: int = 0,
):
    """Evaluate the current policy without exploration noise."""

    number_eval_episodes = int(config.number_eval_episodes)
    state = env.reset()

    for eval_episode_counter in range(number_eval_episodes):
        episode_reward = 0
        done = False
        truncated = False

        while not done and not truncated:
            action = agent.select_action_from_policy(state, evaluation=True)
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


def policy_based_train(
    env,
    agent,
    memory,
    record,
    train_config: TrainingConfig,
    alg_config: AlgorithmConfig,
):
    """Train a policy-based agent with IER episode-reward repetition."""

    logging.info("Policy-based training with IER episode-reward repetition.")

    start_time = time.time()

    explore = False
    repeat_episode = False

    max_steps_training = alg_config.max_steps_training
    max_steps_exploration = alg_config.max_steps_exploration
    number_steps_per_evaluation = train_config.number_steps_per_evaluation
    number_steps_per_train_policy = alg_config.number_steps_per_train_policy

    intrinsic_on = (
        bool(alg_config.intrinsic_on)
        if hasattr(alg_config, "intrinsic_on")
        else False
    )

    min_noise = alg_config.min_noise if hasattr(alg_config, "min_noise") else 0
    noise_decay = alg_config.noise_decay if hasattr(alg_config, "noise_decay") else 1.0
    noise_scale = alg_config.noise_scale if hasattr(alg_config, "noise_scale") else 0.1

    logging.info(
        "Training %s | Exploration %s | Evaluation %s",
        max_steps_training,
        max_steps_exploration,
        number_steps_per_evaluation,
    )

    batch_size = alg_config.batch_size
    gradient_steps = alg_config.G

    episode_timesteps = 0
    episode_reward = 0
    episode_num = 0

    repeat_count = 0
    repeat_total_reward = 0
    repeated_actions = []

    best_episode_reward = 0
    evaluate = False

    # Number of immediate repetitions after discovering a new best episode.
    repetition_number = 2

    state = env.reset()
    episode_start = time.time()

    for total_step_counter in range(int(max_steps_training)):
        episode_timesteps += 1

        if total_step_counter < max_steps_exploration or explore:
            logging.info(
                "Running exploration step %s/%s",
                total_step_counter + 1,
                max_steps_exploration,
            )

            action_env = env.sample_action()
            action = hlp.normalize(
                action_env,
                env.max_action_value,
                env.min_action_value,
            )

        elif repeat_count > 0 and repeat_episode:
            action = repeated_actions[episode_timesteps - 1]
            action_env = hlp.denormalize(
                action,
                env.max_action_value,
                env.min_action_value,
            )

            if episode_timesteps >= len(repeated_actions):
                repeat_episode = False
                logging.info(
                    "Reached end of repeated episode for repetition %s.",
                    repetition_number - repeat_count,
                )

        else:
            noise_scale *= noise_decay
            noise_scale = max(min_noise, noise_scale)

            action = agent.select_action_from_policy(
                state,
                noise_scale=noise_scale,
            )
            action_env = hlp.denormalize(
                action,
                env.max_action_value,
                env.min_action_value,
            )

        next_state, reward_extrinsic, done, truncated = env.step(action_env)

        intrinsic_reward = 0
        if intrinsic_on and total_step_counter > max_steps_exploration:
            intrinsic_reward = agent.get_intrinsic_reward(
                state,
                action,
                next_state,
            )

        total_reward = reward_extrinsic + intrinsic_reward

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

        if (
            total_step_counter >= max_steps_exploration
            and total_step_counter % number_steps_per_train_policy == 0
        ):
            for _ in range(gradient_steps):
                agent.train_policy(memory, batch_size)

        if (total_step_counter + 1) % number_steps_per_evaluation == 0:
            evaluate = True

        if done or truncated:
            logging.info(
                "Episode ended | steps=%s | repeated_actions=%s | repeat_count=%s",
                episode_timesteps,
                len(repeated_actions),
                repeat_count,
            )

            episode_time = time.time() - episode_start

            record.log_train(
                total_steps=total_step_counter + 1,
                episode=episode_num + 1,
                episode_steps=episode_timesteps,
                episode_reward=episode_reward,
                episode_time=episode_time,
                display=True,
            )

            if repeat_count == 1:
                repeat_episode = False
                repeat_count -= 1

                logging.info(
                    "Finished final immediate repetition | repeated_total_reward=%s | "
                    "episode_steps=%s | transition_reward=%s | episode_reward=%s",
                    repeat_total_reward,
                    episode_timesteps,
                    total_reward,
                    episode_reward,
                )

            elif repeat_count > 1:
                repeat_count -= 1
                repeat_episode = True

                logging.info(
                    "Continuing immediate repetition | transition_reward=%s | "
                    "episode_reward=%s",
                    total_reward,
                    episode_reward,
                )

            elif (
                repeat_count == 0
                and episode_reward > 0
                and episode_reward > best_episode_reward
            ):
                logging.info(
                    "New best episode found | episode=%s | episode_reward=%s | "
                    "previous_best=%s",
                    episode_num,
                    episode_reward,
                    best_episode_reward,
                )

                best_episode_reward = episode_reward

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

                repeated_actions = actions
                repeat_count = repetition_number

            if evaluate:
                logging.info("************* Evaluation Loop *************")
                evaluate_policy_network(
                    env,
                    agent,
                    train_config,
                    record=record,
                    total_steps=total_step_counter,
                )
                logging.info("-------------------------------------------")
                evaluate = False

            state = env.reset()
            episode_timesteps = 0
            episode_reward = 0
            episode_num += 1
            episode_start = time.time()

    elapsed_time = time.time() - start_time

    logging.info(
        "Training time: %s",
        time.strftime("%H:%M:%S", time.gmtime(elapsed_time)),
    )