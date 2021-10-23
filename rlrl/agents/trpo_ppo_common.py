import itertools
from typing import Dict, List

import torch
from torch import distributions

from rlrl.replay_buffers import TorchTensorBatch
from rlrl.utils.transpose_list_dict import transpose_list_dict


class TorchTensorBatchTrpoPpo(TorchTensorBatch):
    def __init__(
        self,
        state,
        next_state,
        action,
        reward,
        terminal,
        log_prob,
        v_pred,
        next_v_pred,
        adv,
        v_target,
        device="cpu",
        **kwargs
    ) -> None:
        self.log_prob: torch.Tensor = torch.tensor(log_prob)
        self.v_pred: torch.Tensor = torch.tensor(v_pred)
        self.next_v_pred: torch.Tensor = torch.tensor(next_v_pred)
        self.adv: torch.Tensor = torch.tensor(adv)
        self.v_target: torch.Tensor = torch.tensor(v_target)
        super().__init__(state, next_state, action, reward, terminal, device=device, **kwargs)

    def to(self, device):
        super().to(device=device)
        self.log_prob = self.log_prob.to(device)
        self.v_pred = self.v_pred.to(device)
        self.next_v_pred = self.next_v_pred.to(device)
        self.adv = self.adv.to(device)
        self.v_target = self.v_target.to(device)
        return self

    def __len__(self):
        batch_len = super().__len__()
        assert (
            batch_len
            == len(self.log_prob)
            == len(self.v_pred)
            == len(self.next_v_pred)
            == len(self.adv)
            == len(self.v_target)
        )
        return batch_len


def _add_adv_and_v_target_to_episode(episode, gamma, lambd):
    adv = 0.0
    for transition in reversed(episode):
        td_err = (
            transition["reward"]
            + gamma * (not transition["terminal"]) * transition["next_v_pred"]
            - transition["v_pred"]
        )
        adv = td_err + gamma * lambd * adv
        transition["adv"] = adv
        transition["v_target"] = adv + transition["v_pred"]


def _memory2batch(
    memory: List[List[Dict]],  # [episode, episode,...]
    gamma: float,
    lambd: float,
    vf,
    policy,
    state_normalizer,
    device,
):
    memory_flatten = list(itertools.chain.from_iterable(memory))
    batch_flatten = TorchTensorBatch(**transpose_list_dict(memory_flatten), device=device)

    state = batch_flatten.state
    next_state = batch_flatten.next_state

    if state_normalizer:
        state = state_normalizer(batch_flatten.state, update=False)
        next_state = state_normalizer(batch_flatten.next_state, update=False)

    with torch.no_grad():
        distribs: distributions.Distribution = policy(state)
        vs_pred = vf(state)
        next_vs_pred = vf(next_state)
        log_probs = distribs.log_prob(batch_flatten.action)

    for transition, log_prob, v_pred, next_v_pred in zip(
        memory_flatten, log_probs, vs_pred, next_vs_pred
    ):
        transition["log_prob"] = log_prob
        transition["v_pred"] = v_pred
        transition["next_v_pred"] = next_v_pred

    for episode in memory:
        _add_adv_and_v_target_to_episode(episode, gamma, lambd)

    return TorchTensorBatchTrpoPpo(**transpose_list_dict(memory_flatten), device=device)
