import copy
import logging
from typing import Union, Any, Dict, Type


import torch
import torch.nn.functional as F
from torch import cuda
from torch.optim import Adam, Optimizer

from rl_algos.agents.td3_agent import (
    TD3,
    default_policy_fn,
    default_q_fn,
    default_target_policy_smoothing_func,
)
from rl_algos.buffers import ReplayBuffer, TrainingBatch
from rl_algos.explorers import ExplorerBase, GaussianExplorer
from rl_algos.utils.sync_param import synchronize_parameters
from .rate import default_rate_fn


class ATD3FixedResetCost(TD3):
    saved_attributes = (
        # q function
        "q1",
        "q2",
        "q1_target",
        "q2_target",
        "q1_optimizer",
        "q2_optimizer",
        # rate
        "rate",
        "rate_optimizer",
        "rate_target",
        # policy
        "policy",
        "policy_target",
        "policy_optimizer",
        # replay buffer
        "replay_buffer",
    )

    def __init__(
        self,
        dim_state: int,
        dim_action: int,
        q_fn=default_q_fn,
        policy_fn=default_policy_fn,
        rate_fn=default_rate_fn,
        reset_cost: float = 100,
        policy_update_delay: int = 2,
        policy_smoothing_func=default_target_policy_smoothing_func,
        tau: float = 5e-3,
        explorer: ExplorerBase = GaussianExplorer(0.1, -1, 1),
        replay_buffer: ReplayBuffer = ReplayBuffer(10**6),
        batch_size: int = 256,
        replay_start_size: int = 25e3,
        optimizer_class: Type[Optimizer] = Adam,
        optimizer_kwargs: Dict[str, Any] = {"lr": 1e-3},
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
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            calc_stats=calc_stats,
            logger=logger,
            device=device,
        )
        self.reset_cost = reset_cost

        self.rate = rate_fn().to(self.device)
        self.rate_optimizer = optimizer_class(self.rate.parameters(), **optimizer_kwargs)
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
            if self.num_q_update % self.policy_update_delay == 0:
                self._update_actor(batch)
                self._sync_target_network()

    def _update_critic(self, batch: TrainingBatch):
        q_loss, rate_loss = self.compute_q_and_rate_loss(batch)

        for optimizer in [self.q1_optimizer, self.q2_optimizer, self.rate_optimizer]:
            optimizer.zero_grad()

        for loss in [q_loss, rate_loss]:
            loss.backward()

        for optimizer in [self.q1_optimizer, self.q2_optimizer, self.rate_optimizer]:
            optimizer.step()

        self.num_q_update += 1

    def compute_q_and_rate_loss(self, batch: TrainingBatch):
        with torch.no_grad():
            next_actions: torch.Tensor = self.policy_smoothing_func(
                self.policy_target(batch.next_state).sample()
            )
            next_q1 = torch.flatten(self.q1_target((batch.next_state, next_actions)))
            next_q2 = torch.flatten(self.q2_target((batch.next_state, next_actions)))
            next_q = torch.min(next_q1, next_q2)

            reward = torch.nan_to_num(batch.reward, -self.reset_cost)

            q_target: torch.Tensor = reward - self.rate_target() + next_q

            current_q1 = torch.flatten(self.q1_target((batch.state, batch.action)))
            current_q2 = torch.flatten(self.q2_target((batch.state, batch.action)))
            current_q = (current_q1 + current_q2) / 2.0

            rate_target = reward - current_q + next_q

        q1_pred = torch.flatten(self.q1((batch.state, batch.action)))
        q2_pred = torch.flatten(self.q2((batch.state, batch.action)))

        q_loss = F.mse_loss(q1_pred, q_target) + F.mse_loss(q2_pred, q_target)

        rate_pred = self.rate()

        rate_loss = F.mse_loss(rate_pred, rate_target.mean())

        if self.stats is not None:
            self.stats("q1_pred").extend(q1_pred.detach().cpu().numpy())
            self.stats("q2_pred").extend(q2_pred.detach().cpu().numpy())
            self.stats("q_target").extend(q_target.detach().cpu().numpy())
            self.stats("q_loss").append(q_loss.detach().cpu().numpy())
            self.stats("rate_pred").append(float(rate_pred))
            self.stats("rate_targets", methods=["mean", "var"]).extend(
                rate_target.detach().cpu().numpy()
            )
        return q_loss, rate_loss

    def _sync_target_network(self):
        synchronize_parameters(
            src=self.rate,
            dst=self.rate_target,
            method="soft",
            tau=self.tau,
        )
        return super()._sync_target_network()
