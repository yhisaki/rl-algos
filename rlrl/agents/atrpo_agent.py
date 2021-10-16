import itertools
from typing import Dict, List

import numpy as np
import torch
from torch import distributions

from rlrl.agents import TrpoAgent
from rlrl.agents.trpo_ppo_common import TorchTensorBatchTrpoPpo
from rlrl.nn.z_score_filter import ZScoreFilter  # NOQA
from rlrl.replay_buffers import TorchTensorBatch
from rlrl.utils.transpose_list_dict import transpose_list_dict


def _add_adv_and_v_target_to_episode(episode, rho, lambd):
    adv = 0.0
    for transition in reversed(episode):
        td_err = transition["reward"] - rho + transition["next_v_pred"] - transition["v_pred"]
        adv = td_err + lambd * adv
        transition["adv"] = adv
        transition["v_target"] = adv + transition["v_pred"]


def _memory2batch(
    memory: List[List[Dict]],  # [episode, episode,...]
    vf,
    lambd,
    policy,
    state_normalizer,
    device,
):
    memory_flatten = list(itertools.chain.from_iterable(memory))
    batch_flatten = TorchTensorBatch(**transpose_list_dict(memory_flatten), device=device)

    state = batch_flatten.state
    next_state = batch_flatten.next_state
    rho = batch_flatten.reward.mean()

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
        _add_adv_and_v_target_to_episode(episode, rho, lambd)

    return TorchTensorBatchTrpoPpo(**transpose_list_dict(memory_flatten), device=device)


class AtrpoAgent(TrpoAgent):
    def __init__(
        self,
        *args,
        reset_cost=-100,
        **kwargs,
    ) -> None:
        super().__init__(
            *args,
            gamma=None,
            **kwargs,
        )
        self.reset_cost = np.float32(reset_cost)
        self.episode_min_step = 1000
        self.logger.info(f"Atrpo uses {self.device}")

    def _memory_preprocessing(self, memory: List[List[Dict]]):
        return _memory2batch(
            memory,
            self.vf,
            self.lambd,
            self.policy,
            self.state_normalizer,
            device=self.device,
        )

    def update_if_dataset_is_ready(self):
        assert self.training
        self.just_updated = False
        memory_size = sum([len(episode) for episode in itertools.chain.from_iterable(self.memory)])
        if memory_size >= self.update_interval:
            self.just_updated = True
            self.num_update += 1
            self.logger.info(f"Update TRPO num: {self.num_update}")

            def expand_episodes(episodes):
                for episode in episodes:
                    for transition in episode:
                        if transition["terminal"]:
                            transition["reward"] += self.reset_cost
                        yield transition

            expanded_memory = []
            for episodes in self.memory:
                expanded_memory.append(list(expand_episodes(episodes)))

            batch = self._memory_preprocessing(expanded_memory)

            self.logger.info(f"rho = {float(batch.reward.mean())}")

            if self.state_normalizer:
                self.state_normalizer.update(batch.state)

            self._update_vf(batch)
            self._update_policy(batch)

            self.memory = [[[]] for _ in range(self.num_envs)]  # reset memory
