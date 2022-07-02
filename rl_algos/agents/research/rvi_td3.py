import logging
from typing import Union

import torch
import torch.nn.functional as F
from torch import cuda

from rl_algos.agents.td3_agent import (
    TD3,
    NetworkAndOptimizerFunc,
    default_policy_fn,
    default_q_fn,
    default_target_policy_smoothing_func,
)
from rl_algos.buffers import ReplayBuffer, TrainingBatch
from rl_algos.explorers import ExplorerBase, GaussianExplorer


class RVI_TD3(TD3):
    def __init__(
        self,
        dim_state: int,
        dim_action: int,
        q_fn: NetworkAndOptimizerFunc = default_q_fn,
        policy_fn: NetworkAndOptimizerFunc = default_policy_fn,
        policy_update_delay: int = 2,
        policy_smoothing_func=default_target_policy_smoothing_func,
        tau: float = 5e-3,
        explorer: ExplorerBase = GaussianExplorer(0.1, -1, 1),
        reset_cost: float = 100,
        replay_buffer: ReplayBuffer = ReplayBuffer(10**6),
        batch_size: int = 256,
        replay_start_size: int = 25e3,
        calc_stats: bool = True,
        logger: logging.Logger = logging.getLogger(__name__),
        device: Union[str, torch.device] = torch.device("cuda:0" if cuda.is_available() else "cpu"),
    ) -> None:
        super().__init__(
            dim_state=dim_state,
            dim_action=dim_action,
            q_fn=q_fn,
            policy_fn=policy_fn,
            policy_update_delay=policy_update_delay,
            policy_smoothing_func=policy_smoothing_func,
            tau=tau,
            explorer=explorer,
            gamma=None,
            replay_buffer=replay_buffer,
            batch_size=batch_size,
            replay_start_size=replay_start_size,
            calc_stats=calc_stats,
            logger=logger,
            device=device,
        )

        self.reset_cost = reset_cost

    def compute_q_loss(self, batch: TrainingBatch):
        with torch.no_grad():
            fq = torch.mean(
                self.q1_target((batch.state, batch.action))
                + self.q2_target((batch.state, batch.action))
            )
            next_actions: torch.Tensor = self.policy_smoothing_func(
                self.policy_target(batch.next_state).sample()
            )
            next_q1 = torch.flatten(self.q1_target((batch.next_state, next_actions)))
            next_q2 = torch.flatten(self.q2_target((batch.next_state, next_actions)))

            q_target: torch.Tensor = (
                torch.nan_to_num(batch.reward, -self.reset_cost) - fq + torch.min(next_q1, next_q2)
            )
        q1_pred = torch.flatten(self.q1((batch.state, batch.action)))
        q2_pred = torch.flatten(self.q2((batch.state, batch.action)))

        loss = F.mse_loss(q1_pred, q_target) + F.mse_loss(q2_pred, q_target)

        if self.stats is not None:
            self.stats("fq").append(fq.detach().cpu().numpy())
            self.stats("q1_pred").extend(q1_pred.detach().cpu().numpy())
            self.stats("q2_pred").extend(q2_pred.detach().cpu().numpy())
            self.stats("q_target").extend(q_target.detach().cpu().numpy())
            self.stats("q_loss").append(loss.detach().cpu().numpy())
        return loss
