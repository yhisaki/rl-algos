import copy
import logging
from typing import Union

import torch
import torch.nn.functional as F
from torch import cuda

from rl_algos.agents.td3_agent import (
    TD3,
    default_policy_fn,
    default_q_fn,
    default_target_policy_smoothing_func,
)
from rl_algos.buffers import ReplayBuffer, TrainingBatch
from rl_algos.explorers import ExplorerBase, GaussianExplorer
from rl_algos.utils.sync_param import synchronize_parameters

from .reset_cost import default_reset_cost_fn
from .rate import default_rate_initializer


class ATD3(TD3):
    def __init__(
        self,
        dim_state: int,
        dim_action: int,
        q_fn=default_q_fn,
        policy_fn=default_policy_fn,
        rate_fn=default_rate_initializer,
        policy_update_delay: int = 2,
        policy_smoothing_func=default_target_policy_smoothing_func,
        tau: float = 5e-3,
        explorer: ExplorerBase = GaussianExplorer(0.1, -1, 1),
        reset_cost_fn=default_reset_cost_fn,
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

        self.reset_cost, self.reset_cost_optimizer = reset_cost_fn(self.device)
        self.saved_attributes += ("reset_cost", "reset_cost_optimizer")
        self.target_terminal_probability = target_terminal_probability

        self.rate, self.rate_optimizer = rate_fn(device)
        self.rate_target = copy.deepcopy(self.rate).eval().requires_grad_(False)

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
            self._update_reset_cost(batch)

            if self.num_q_update % self.policy_update_delay == 0:
                self._update_actor(batch)
                self._sync_target_network()

    def _update_reset_cost(self, batch: TrainingBatch):
        reset_cost_loss = self.compute_reset_cost_loss(batch)
        self.reset_cost_optimizer.zero_grad()
        reset_cost_loss.backward()
        self.reset_cost_optimizer.step()

    def _update_critic(self, batch: TrainingBatch):
        super()._update_critic(batch)
        self._update_rate(batch)

    def _update_rate(self, batch: TrainingBatch):
        rate_loss = self.compute_rate_loss(batch)
        self.rate_optimizer.zero_grad()
        rate_loss.backward()
        self.rate_optimizer.step()

    def compute_reset_cost_loss(self, batch: TrainingBatch):
        reset_cost = self.reset_cost()
        loss = -torch.mean(
            reset_cost * (torch.isnan(batch.reward).float() - self.target_terminal_probability)
        )

        if self.stats is not None:
            self.stats("reset_cost").append(float(reset_cost))
            self.stats("reset_cost_loss").append(float(loss))

        return loss

    def compute_rate_loss(self, batch: TrainingBatch):
        with torch.no_grad():
            next_actions: torch.Tensor = self.policy_smoothing_func(
                self.policy_target(batch.next_state).sample()
            )
            next_q1 = torch.flatten(self.q1_target((batch.next_state, next_actions)))
            next_q2 = torch.flatten(self.q2_target((batch.next_state, next_actions)))
            next_q = torch.min(next_q1, next_q2)

            current_q1 = torch.flatten(self.q1_target((batch.state, batch.action)))
            current_q2 = torch.flatten(self.q2_target((batch.state, batch.action)))

            current_q = (current_q1 + current_q2) / 2.0

            rate_targets = (
                torch.nan_to_num(batch.reward, -float(self.reset_cost())) - current_q + next_q
            )

        rate_pred: torch.Tensor = self.rate()
        rate_loss = F.mse_loss(rate_pred, rate_targets.mean())
        if self.stats is not None:
            self.stats("rate_pred").append(float(rate_pred))
            self.stats("rate_targets", methods=["mean", "var"]).extend(
                rate_targets.detach().cpu().numpy()
            )
        return rate_loss

    def compute_q_loss(self, batch: TrainingBatch):
        with torch.no_grad():

            next_actions: torch.Tensor = self.policy_smoothing_func(
                self.policy_target(batch.next_state).sample()
            )
            next_q1 = torch.flatten(self.q1_target((batch.next_state, next_actions)))
            next_q2 = torch.flatten(self.q2_target((batch.next_state, next_actions)))

            q_target: torch.Tensor = (
                torch.nan_to_num(batch.reward, -float(self.reset_cost()))
                - self.rate_target()
                + torch.min(next_q1, next_q2)
            )
        q1_pred = torch.flatten(self.q1((batch.state, batch.action)))
        q2_pred = torch.flatten(self.q2((batch.state, batch.action)))

        loss = F.mse_loss(q1_pred, q_target) + F.mse_loss(q2_pred, q_target)

        if self.stats is not None:
            self.stats("q1_pred").extend(q1_pred.detach().cpu().numpy())
            self.stats("q2_pred").extend(q2_pred.detach().cpu().numpy())
            self.stats("q_target").extend(q_target.detach().cpu().numpy())
            self.stats("q_loss").append(loss.detach().cpu().numpy())
        return loss

    def _sync_target_network(self):
        synchronize_parameters(
            src=self.rate,
            dst=self.rate_target,
            method="soft",
            tau=self.tau,
        )
        return super()._sync_target_network()
