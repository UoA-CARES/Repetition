"""
General utility helpers for Repetition-RL.
"""

import os
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch


def create_path_from_format_string(
    format_str: str,
    algorithm: str,
    domain: str,
    task: str,
    gym: str,
    seed: int,
    run_name: str,
) -> str:
    base_dir = os.environ.get(
        "REPETITION_RL_LOG_BASE_DIR",
        f"{Path.home()}/repetition_rl_logs",
    )

    domain_with_hyphen = f"{domain}-" if domain else ""
    domain_task = domain_with_hyphen + task
    date = datetime.now().strftime("%y_%m_%d_%H-%M-%S")

    log_dir = format_str.format(
        algorithm=algorithm,
        domain=domain,
        task=task,
        gym=gym,
        run_name=run_name if run_name else "unnamed",
        run_name_else_date=run_name if run_name else date,
        seed=seed,
        domain_task=domain_task,
        date=date,
    )

    return f"{base_dir}/{log_dir}"


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def soft_update_params(net, target_net, tau: float) -> None:
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


def weight_init(module: torch.nn.Module) -> None:
    if isinstance(module, torch.nn.Linear):
        torch.nn.init.orthogonal_(module.weight.data)
        module.bias.data.fill_(0.0)

    elif isinstance(module, (torch.nn.Conv2d, torch.nn.ConvTranspose2d)):
        assert module.weight.size(2) == module.weight.size(3)
        module.weight.data.fill_(0.0)
        module.bias.data.fill_(0.0)

        mid = module.weight.size(2) // 2
        gain = torch.nn.init.calculate_gain("relu")
        torch.nn.init.orthogonal_(module.weight.data[:, :, mid, mid], gain)


def normalize_observation(observation: torch.Tensor, statistics: dict) -> torch.Tensor:
    return (observation - statistics["observation_mean"]) / statistics[
        "observation_std"
    ]


def normalize_observation_delta(delta: torch.Tensor, statistics: dict) -> torch.Tensor:
    return (delta - statistics["delta_mean"]) / statistics["delta_std"]


def denormalize_observation_delta(
    normalized_delta: torch.Tensor,
    statistics: dict,
) -> torch.Tensor:
    return (normalized_delta * statistics["delta_std"]) + statistics["delta_mean"]


def denormalize(action, max_action_value, min_action_value):
    max_range_value = max_action_value
    min_range_value = min_action_value
    max_value_in = 1
    min_value_in = -1

    return (action - min_value_in) * (
        max_range_value - min_range_value
    ) / (max_value_in - min_value_in) + min_range_value


def normalize(action, max_action_value, min_action_value):
    max_range_value = 1
    min_range_value = -1
    max_value_in = max_action_value
    min_value_in = min_action_value

    return (action - min_value_in) * (
        max_range_value - min_range_value
    ) / (max_value_in - min_value_in) + min_range_value


def compare_models(model_1: torch.nn.Module, model_2: torch.nn.Module) -> bool:
    models_differ = 0

    for key_item_1, key_item_2 in zip(
        model_1.state_dict().items(),
        model_2.state_dict().items(),
    ):
        if not torch.equal(key_item_1[1], key_item_2[1]):
            models_differ += 1

            if key_item_1[0] == key_item_2[0]:
                print("Mismatch found at", key_item_1[0])
            else:
                raise ValueError(
                    f"Models are not equal. {key_item_1[0]} is not equal to {key_item_2[0]}"
                )

    return models_differ == 0


def prioritized_approximate_loss(
    x: torch.Tensor,
    min_priority: float,
    alpha: float,
) -> torch.Tensor:
    return torch.where(
        x.abs() < min_priority,
        (min_priority**alpha) * 0.5 * x.pow(2),
        min_priority * x.abs().pow(1.0 + alpha) / (1.0 + alpha),
    ).mean()


def huber(x: torch.Tensor, min_priority: float) -> torch.Tensor:
    return torch.where(
        x < min_priority,
        0.5 * x.pow(2),
        min_priority * x,
    ).mean()


def quantile_huber_loss_f(
    quantiles: torch.Tensor,
    samples: torch.Tensor,
) -> torch.Tensor:
    pairwise_delta = samples[:, None, None, :] - quantiles[:, :, :, None]
    abs_pairwise_delta = torch.abs(pairwise_delta)

    huber_loss = torch.where(
        abs_pairwise_delta > 1,
        abs_pairwise_delta - 0.5,
        pairwise_delta**2 * 0.5,
    )

    n_quantiles = quantiles.shape[2]
    tau = (
        torch.arange(n_quantiles, device=pairwise_delta.device).float() / n_quantiles
        + 1 / 2 / n_quantiles
    )

    loss = (
        torch.abs(tau[None, None, :, None] - (pairwise_delta < 0).float())
        * huber_loss
    ).mean()

    return loss


def flatten(w: int, k: int = 3, s: int = 1, p: int = 0, m: bool = True) -> int:
    return int((np.floor((w - k + 2 * p) / s) + 1) if m else 1)