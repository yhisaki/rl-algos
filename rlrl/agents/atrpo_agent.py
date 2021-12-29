import logging
from typing import Optional, Type, Union

import torch
from torch import cuda, nn
from torch.optim import Adam, Optimizer

from rlrl.agents.trpo_agent import TrpoAgent
from rlrl.buffers import EpisodicTrainingBatch
from rlrl.modules.z_score_filter import ZScoreFilter


def average_version_generalized_advantage_estimation(
    batch: EpisodicTrainingBatch, lambd: float, vf: nn.Module, device
):
    with torch.no_grad():
        rho = torch.mean(batch.flatten.reward)
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
                td_err = transition.reward - rho + next_v_pred - v_pred
                adv = td_err + lambd * adv
                advantages_per_episode.insert(0, adv)
                v_targets_per_episode.insert(0, adv + v_pred)
            advantages.extend(advantages_per_episode)
            v_targets.extend(v_targets_per_episode)
        return torch.tensor(advantages, device=device), torch.tensor(v_targets, device=device)


class AtrpoAgent(TrpoAgent):
    def __init__(
        self,
        dim_state: int,
        dim_action: int,
        policy: Optional[nn.Module] = None,
        vf: Optional[nn.Module] = None,
        vf_optimizer_class: Type[Optimizer] = Adam,
        vf_optimizer_kwargs: dict = {},
        vf_epoch=3,
        vf_batch_size=64,
        update_interval: int = 5000,
        recurrent: bool = False,
        state_normalizer: Optional[ZScoreFilter] = None,
        lambd: float = 0.97,
        entropy_coef: float = 0.01,
        max_kl: float = 0.01,
        line_search_max_backtrack: int = 10,
        conjugate_gradient_max_iter: int = 10,
        conjugate_gradient_damping=0.01,
        device: Union[str, torch.device] = torch.device("cuda:0" if cuda.is_available() else "cpu"),
        calc_stats=True,
        value_stats_window=None,
        entropy_stats_window=None,
        kl_stats_window=None,
        logger: logging.Logger = logging.getLogger(__name__),
    ) -> None:
        super().__init__(
            dim_state,
            dim_action,
            policy=policy,
            vf=vf,
            vf_optimizer_class=vf_optimizer_class,
            vf_optimizer_kwargs=vf_optimizer_kwargs,
            vf_epoch=vf_epoch,
            vf_batch_size=vf_batch_size,
            update_interval=update_interval,
            recurrent=recurrent,
            state_normalizer=state_normalizer,
            gamma=None,
            lambd=lambd,
            entropy_coef=entropy_coef,
            max_kl=max_kl,
            line_search_max_backtrack=line_search_max_backtrack,
            conjugate_gradient_max_iter=conjugate_gradient_max_iter,
            conjugate_gradient_damping=conjugate_gradient_damping,
            device=device,
            calc_stats=calc_stats,
            value_stats_window=value_stats_window,
            entropy_stats_window=entropy_stats_window,
            kl_stats_window=kl_stats_window,
            logger=logger,
        )

    def update_if_dataset_is_ready(self):
        assert self.training
        self.just_updated = False
        if len(self.buffer) >= self.update_interval:
            self.just_updated = True
            self.num_update += 1
            self.logger.info(f"Update ATRPO num: {self.num_update}")

            episodes = self.buffer.get_episodes()
            batch = EpisodicTrainingBatch(episodes, device=self.device)

            if self.state_normalizer:
                batch.flatten.state = self.state_normalizer(batch.flatten.state, update=True)
                batch.flatten.next_state = self.state_normalizer(
                    batch.flatten.next_state, update=False
                )

            adv, v_target = average_version_generalized_advantage_estimation(
                batch, lambd=self.lambd, vf=self.vf, device=self.device
            )
            self._update_policy(batch.flatten, adv)
            self._update_vf(batch.flatten, v_target)
            self.buffer.clear()
