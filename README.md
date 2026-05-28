# Repetition-RL

<p align="center">
  <img src="assets/exploration-explotation-repetition.png" width="90%">
</p>

<p align="center">
  <b>Repetition as a Third Mode of Interaction in Reinforcement Learning</b>
</p>

---

## Overview

`Repetition-RL` is a research framework for studying **repetition as a third mode of interaction** in RL.

Standard RL agents usually interact with the environment through two main modes:

1. **Exploration**: trying actions to discover useful behaviours.
2. **Exploitation**: using the current policy to maximise return.

This repository adds a third interaction mode:

3. **Repetition**: deliberately re-executing previously valuable action sequences during environment interaction.

The key idea is that valuable behaviour should not only be reused inside the replay buffer during optimisation. It can also be reused directly in the environment by repeating trajectories that were useful before.

This repository supports:

- Standard TD3 and SAC
- SIL variants
- Repetition-based TD3 and SAC
- Instant Episode Repetition
- Spaced Episode Repetition
- Episode-reward, transition-reward, and mixed selection strategies

---

# Research Paper

This repository is associated with the journal paper:

> **Beyond Exploration and Exploitation: Repetition as a Third Mode of Interaction in Reinforcement Learning**

Citation placeholder:

```bibtex
@article{yamani2026repetition,
  title={Beyond Exploration and Exploitation: Repetition as a Third Mode of Interaction in Reinforcement Learning},
  author={Yamani, Hoda and MacDonald, Bruce and Williams, Henry},
  journal={Under Review},
  year={2026}
}
```

---

# Core Idea

<p align="center">
  <img src="assets/exploration-explotation-repetition.png" width="90%">
</p>

In standard off-policy RL, past experiences are reused through sampling from a replay buffer. This helps the optimisation process, but it does not directly affect how the agent behaves during environment interaction.

In contrast, repetition modifies the interaction process itself.

Instead of only asking:

> Which transition should be sampled for training?

Repetition also asks:

> Which valuable behaviour should the agent re-execute in the environment?

This makes repetition different from replay-buffer sampling. Replay is an optimisation mechanism. Repetition is an interaction mechanism.

---

# Methods

## 1. Instant Episode Repetition (IER)

<p align="center">
  <img src="assets/IER.png" width="80%">
</p>

IER repeats a valuable episode soon after it is discovered.

The agent first interacts normally with the environment. When a useful trajectory is detected, the algorithm stores the corresponding action sequence and immediately re-executes it for a small number of repetitions.

### Why IER?

IER is useful when a good behaviour appears briefly and may be difficult to rediscover. Instead of waiting for the replay buffer to indirectly reinforce it, IER repeats the behaviour directly.

### IER selection strategies

This repository supports two IER selection strategies:

#### IER with episode reward

The episode is selected based on total episode return. This is useful when the whole trajectory is globally successful.

Run:

```bash
REPETITION_FRAMEWORK=IER \
SELECTION_STRATEGY=episode_reward \
python3 train.py run --gym openai --task HalfCheetah-v4 ReTD3
```

#### IER with transition reward

The episode is selected based on a high local transition reward. This is useful when a locally important event happens inside an episode, even if the full episode return is not the best.

Run:

```bash
REPETITION_FRAMEWORK=IER \
SELECTION_STRATEGY=transition_reward \
python3 train.py run --gym openai --task HalfCheetah-v4 ReTD3
```

---

## 2. Spaced Episode Repetition (SER)

<p align="center">
  <img src="assets/SER.png" width="85%">
</p>

SER repeats valuable episodes after a delay rather than immediately.

SER uses a long-term episodic memory called:

> **Virtual Episode Memory (VEM)**

The VEM stores valuable episodes and periodically provides action sequences for repetition.

### Why SER?

IER repeats an episode immediately. SER introduces spacing between discovery and repetition. This is inspired by the idea that delayed reuse can help consolidate useful behaviours over time and prevent the agent from relying only on very recent experience.

---

# SER Variants

## ESER: Episode-based SER

ESER selects episodes according to cumulative episode reward.

```math
R_i^{ep} > R_{max}^{ep}
```

This means that complete episodes with high return are stored in VEM and later repeated.

### When ESER is useful

ESER is useful when success depends on the full trajectory rather than one isolated transition.

Example command:

```bash
REPETITION_FRAMEWORK=SER \
SELECTION_STRATEGY=episode_reward \
python3 train.py run --gym openai --task HalfCheetah-v4 ReTD3
```

---

## XSER: Transition-based SER

XSER selects episodes according to high local transition reward.

<p align="center">
  <img src="assets/xser_equation.png" width="45%">
</p>

Instead of using only total episode reward, XSER stores episodes that contain a locally valuable transition.

### When XSER is useful

XSER is useful when important learning signals are sparse or local. For example, a useful contact, movement, or reward event may occur inside an otherwise weak episode.

Example command:

```bash
REPETITION_FRAMEWORK=SER \
SELECTION_STRATEGY=transition_reward \
python3 train.py run --gym openai --task HalfCheetah-v4 ReTD3
```

---

## MSER: Mixed SER

MSER combines both ESER and XSER.

<p align="center">
  <img src="assets/mser_equation.png" width="60%">
</p>

MSER stores and repeats episodes selected by:

- global episode reward
- local transition reward

This allows the agent to preserve both globally successful behaviours and locally important experiences.

Example command:

```bash
REPETITION_FRAMEWORK=SER \
SELECTION_STRATEGY=mixed \
python3 train.py run --gym openai --task HalfCheetah-v4 ReTD3
```

A simple mixed version is also supported:

```bash
REPETITION_FRAMEWORK=SER \
SELECTION_STRATEGY=simple_mixed \
python3 train.py run --gym openai --task HalfCheetah-v4 ReTD3
```

---

# Repository Structure

```text
repetition-rl/
│
├── algorithms/
│   ├── base/
│   │   ├── td3.py
│   │   └── sac.py
│   │
│   ├── repetition/
│   │   ├── retd3.py
│   │   └── resac.py
│   │
│   └── sil/
│       ├── sil_td3.py
│       └── sil_sac.py
│
├── networks/
│   ├── td3_networks.py
│   ├── sac_networks.py
│   └── network_factory.py
│
├── memory/
│   ├── replay_buffer.py
│   ├── episodic_memory.py
│   ├── prioritized_replay_buffer.py
│   ├── sum_tree.py
│   └── memory_factory.py
│
├── train_loops/
│   ├── base/
│   │   └── policy_loop.py
│   │
│   ├── ier/
│   │   ├── episode_reward_loop.py
│   │   └── transition_reward_loop.py
│   │
│   └── ser/
│       ├── eser/
│       │   └── episode_reward_loop.py
│       │
│       ├── xser/
│       │   └── transition_reward_loop.py
│       │
│       └── mser/
│           ├── mixed_reward_loop.py
│           └── simple_mixed_reward_loop.py
│
├── environments/
│   ├── mujoco/
│   ├── dm_control/
│   └── wrappers/
│
├── utils/
│   ├── configurations.py
│   ├── helpers.py
│   ├── record.py
│   └── plotter.py
│
├── assets/
├── results/
├── analysis/
├── docs/
├── train.py
└── requirements.txt
```

---

# Supported Algorithms

## Standard RL

| Algorithm | Description |
|---|---|
| TD3 | Twin Delayed Deep Deterministic Policy Gradient |
| SAC | Soft Actor-Critic |

## Self-Imitation Learning

| Algorithm | Description |
|---|---|
| TD3SIL | TD3 with self-imitation learning |
| SACSIL | SAC with self-imitation learning |

## Repetition-Based RL

| Algorithm | Description |
|---|---|
| ReTD3 | TD3 with repetition-based interaction |
| ReSAC | SAC with repetition-based interaction |

---

# Supported Repetition Modes

| Framework | Strategy | Method |
|---|---|---|
| IER | episode_reward | IER with episode reward selection |
| IER | transition_reward | IER with transition reward selection |
| SER | episode_reward | ESER |
| SER | transition_reward | XSER |
| SER | mixed | MSER |
| SER | simple_mixed | Simple MSER |

---

# Installation

Clone the repository:

```bash
git clone https://github.com/UoA-CARES/repetition-rl.git
cd repetition-rl
```

Create an environment:

```bash
conda create -n repetition-rl python=3.10
conda activate repetition-rl
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Running Experiments

The main entry point is:

```bash
python3 train.py run --gym <environment_backend> --task <task_name> <algorithm>
```

Example:

```bash
python3 train.py run --gym openai --task HalfCheetah-v4 TD3
```

---

# Standard Training Commands

## TD3

```bash
python3 train.py run \
--seeds 10 \
--gym openai \
--task HalfCheetah-v4 \
TD3
```

## SAC

```bash
python3 train.py run \
--seeds 10 \
--gym openai \
--task HalfCheetah-v4 \
SAC
```

---

# SIL Training Commands

## TD3SIL

```bash
python3 train.py run \
--seeds 10 \
--gym openai \
--task HalfCheetah-v4 \
TD3SIL
```

## SACSIL

```bash
python3 train.py run \
--seeds 10 \
--gym openai \
--task HalfCheetah-v4 \
SACSIL
```

---

# IER Training Commands

## IER with TD3

```bash
REPETITION_FRAMEWORK=IER \
SELECTION_STRATEGY=episode_reward \
python3 train.py run \
--seeds 10 \
--gym openai \
--task HalfCheetah-v4 \
ReTD3
```

## IER with SAC

```bash
REPETITION_FRAMEWORK=IER \
SELECTION_STRATEGY=episode_reward \
python3 train.py run \
--seeds 10 \
--gym openai \
--task HalfCheetah-v4 \
ReSAC
```

## IER transition reward

```bash
REPETITION_FRAMEWORK=IER \
SELECTION_STRATEGY=transition_reward \
python3 train.py run \
--seeds 10 \
--gym openai \
--task HalfCheetah-v4 \
ReTD3
```

---

# SER Training Commands

## ESER

```bash
REPETITION_FRAMEWORK=SER \
SELECTION_STRATEGY=episode_reward \
python3 train.py run \
--seeds 10 \
--gym openai \
--task HalfCheetah-v4 \
ReTD3
```

## XSER

```bash
REPETITION_FRAMEWORK=SER \
SELECTION_STRATEGY=transition_reward \
python3 train.py run \
--seeds 10 \
--gym openai \
--task HalfCheetah-v4 \
ReTD3
```

## MSER

```bash
REPETITION_FRAMEWORK=SER \
SELECTION_STRATEGY=mixed \
python3 train.py run \
--seeds 10 \
--gym openai \
--task HalfCheetah-v4 \
ReTD3
```

## Simple MSER

```bash
REPETITION_FRAMEWORK=SER \
SELECTION_STRATEGY=simple_mixed \
python3 train.py run \
--seeds 10 \
--gym openai \
--task HalfCheetah-v4 \
ReTD3
```

---

# Multiple Seeds

```bash
python3 train.py run \
--seeds 10 20 30 40 50 \
--gym openai \
--task HalfCheetah-v4 \
TD3
```

---

# Quick Debug Runs

Use `Pendulum-v1` for quick testing:

```bash
python3 train.py run \
--seeds 10 \
--gym openai \
--task Pendulum-v1 \
TD3
```

IER debug:

```bash
REPETITION_FRAMEWORK=IER \
SELECTION_STRATEGY=episode_reward \
python3 train.py run \
--seeds 10 \
--gym openai \
--task Pendulum-v1 \
ReTD3
```

SER debug:

```bash
REPETITION_FRAMEWORK=SER \
SELECTION_STRATEGY=mixed \
python3 train.py run \
--seeds 10 \
--gym openai \
--task Pendulum-v1 \
ReTD3
```

---

# YAML Configuration Files

The current runner mainly uses command-line arguments and configuration classes in:

```text
utils/configurations.py
```

However, YAML files can be added to make experiments easier to reproduce.

Recommended structure:

```text
configs/
├── standard/
│   ├── td3_halfcheetah.yaml
│   └── sac_halfcheetah.yaml
│
├── ier/
│   ├── retd3_episode_reward.yaml
│   └── retd3_transition_reward.yaml
│
└── ser/
    ├── eser_retd3.yaml
    ├── xser_retd3.yaml
    ├── mser_retd3.yaml
    └── simple_mser_retd3.yaml
```

---

## Example YAML: Standard TD3

```yaml
experiment:
  name: td3_halfcheetah
  framework: NONE
  selection_strategy: none

environment:
  gym: openai
  task: HalfCheetah-v4
  domain: ""

algorithm:
  name: TD3
  actor_lr: 0.0003
  critic_lr: 0.0003
  gamma: 0.99
  tau: 0.005

training:
  seeds: [10, 20, 30, 40, 50]
  max_steps_training: 1000000
  max_steps_exploration: 1000
  batch_size: 256
  gradient_steps: 1
  number_steps_per_evaluation: 10000
  number_eval_episodes: 10
```

# Recommended Future YAML Runner

Later, the training command can be simplified to:

```bash
python3 train.py run --config configs/ser/mser_retd3.yaml
```

A small YAML parser can read:

- environment settings
- algorithm settings
- training settings
- repetition settings

and set the correct `REPETITION_FRAMEWORK` and `SELECTION_STRATEGY` automatically.

---

# Output Logs

Experiment outputs are saved under:

```text
~/repetition_rl_logs/
```

Each run stores:

```text
env_config.json
train_config.json
alg_config.json
data/train.csv
data/eval.csv
models/
figures/
videos/
```



# License

MIT License
