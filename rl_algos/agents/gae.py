import torch
from torch import nn

from rl_algos.buffers import EpisodicTrainingBatch


def generalized_advantage_estimation(
    batch: EpisodicTrainingBatch, gamma: float, lambd: float, vf: nn.Module, device
):
    with torch.no_grad():
        advantages = []
        v_targets = []
        for episode in batch:
            advantages_per_episode = []
            v_targets_per_episode = []
            v_preds = vf(episode.state)
            next_v_preds = vf(episode.next_state)
            adv = 0.0
            for transition, v_pred, next_v_pred in zip(
                reversed(episode), reversed(v_preds), reversed(next_v_preds)
            ):
                td_err = (
                    transition.reward
                    + gamma * transition.terminal.logical_not() * next_v_pred
                    - v_pred
                )
                adv = td_err + gamma * lambd * adv
                advantages_per_episode.insert(0, adv)
                v_targets_per_episode.insert(0, adv + v_pred)
            advantages.extend(advantages_per_episode)
            v_targets.extend(v_targets_per_episode)
        return torch.tensor(advantages, device=device), torch.tensor(v_targets, device=device)
