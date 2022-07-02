import logging
from typing import Optional, Union

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

from .reset_cost import ResetCostHolder, default_reset_cost_initializer


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
        reset_cost: Optional[float] = None,
        target_terminal_probability: float = 1 / 256,
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

        self.reset_cost, self.reset_cost_optimizer = default_reset_cost_initializer(
            reset_cost, self.device
        )
        self.saved_attributes += ("reset_cost", "reset_cost_optimizer")
        self.target_terminal_probability = target_terminal_probability

    def update_if_dataset_is_ready(self):
        assert self.training
        self.just_updated = False
        if len(self.replay_buffer) > self.replay_start_size:
            self.just_updated = True
            if self.num_q_update == 0:
                self.logger.info("Start Update")
            sampled = self.replay_buffer.sample(self.batch_size)
            batch = TrainingBatch(**sampled, device=self.device)
            self._update_critic(batch)
            if isinstance(self.reset_cost, ResetCostHolder):
                self._update_reset_cost(batch)

            if self.num_q_update % self.policy_update_delay == 0:
                self._update_actor(batch)
                self._sync_target_network()

    def compute_reset_cost_loss(self, batch: TrainingBatch):
        reset_cost = self.reset_cost()
        loss = -torch.mean(
            reset_cost * (torch.isnan(batch.reward).float() - self.target_terminal_probability)
        )

        if self.stats is not None:
            self.stats("reset_cost").append(float(reset_cost))
            self.stats("reset_cost_loss").append(float(loss))

        return loss

    def _update_reset_cost(self, batch: TrainingBatch):
        reset_cost_loss = self.compute_reset_cost_loss(batch)
        self.reset_cost_optimizer.zero_grad()
        reset_cost_loss.backward()
        self.reset_cost_optimizer.step()

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
                torch.nan_to_num(batch.reward, -float(self.reset_cost()))
                - fq
                + torch.min(next_q1, next_q2)
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
