import copy
import logging
from typing import Union, Any, Dict, Type


import torch
import torch.nn.functional as F
from torch import cuda, nn
from torch.optim import Adam, Optimizer
import rl_algos

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
from .rate import default_rate_fn


class ResetRate(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.reset_rate_param = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

    def forward(self):
        return self.reset_rate_param

    def clip(self):
        self.reset_rate_param.data = torch.clip(self.reset_rate_param, min=0.0, max=1.0)


def default_reset_q_fn(dim_state, dim_action):
    net = nn.Sequential(
        rl_algos.modules.ConcatStateAction(),
        (nn.Linear(dim_state + dim_action, 64)),
        nn.ReLU(),
        (nn.Linear(64, 64)),
        nn.ReLU(),
        (nn.Linear(64, 1)),
    )

    return net


class ATD3(TD3):
    def __init__(
        self,
        dim_state: int,
        dim_action: int,
        q_fn=default_q_fn,
        policy_fn=default_policy_fn,
        rate_fn=default_rate_fn,
        policy_update_delay: int = 2,
        policy_smoothing_func=default_target_policy_smoothing_func,
        tau: float = 5e-3,
        explorer: ExplorerBase = GaussianExplorer(0.1, -1, 1),
        reset_cost_fn=default_reset_cost_fn,
        target_terminal_probability: float = 1 / 256,
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

        self.reset_rate = ResetRate().to(self.device)
        self.reset_rate_target = copy.deepcopy(self.reset_rate).eval().requires_grad_(False)
        self.reset_rate_optimizer = optimizer_class(
            self.reset_rate.parameters(), **optimizer_kwargs
        )

        self.reset_q = default_reset_q_fn(self.dim_state, self.dim_action).to(self.device)
        self.reset_q_target = copy.deepcopy(self.reset_q).eval().requires_grad_(False)
        self.reset_q_optimizer = optimizer_class(self.reset_q.parameters(), **optimizer_kwargs)

        self.reset_cost = reset_cost_fn().to(self.device)
        self.reset_cost_optimizer = optimizer_class(
            self.reset_cost.parameters(), **optimizer_kwargs
        )
        self.saved_attributes += ("reset_cost", "reset_cost_optimizer")
        self.target_terminal_probability = target_terminal_probability

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
            self._update_reset_cost(batch)

            if self.num_q_update % self.policy_update_delay == 0:
                self._update_actor(batch)
                self._sync_target_network()

    def _update_reset_cost(self, batch: TrainingBatch):
        reset_q_loss, reset_rate_loss, reset_cost_loss = self.compute_reset_losses(batch)
        for optimizer in [
            self.reset_q_optimizer,
            self.reset_rate_optimizer,
            self.reset_cost_optimizer,
        ]:
            optimizer.zero_grad()

        for loss in [reset_q_loss, reset_rate_loss, reset_cost_loss]:
            loss.backward()

        for optimizer in [
            self.reset_q_optimizer,
            self.reset_rate_optimizer,
            self.reset_cost_optimizer,
        ]:
            optimizer.step()

        self.reset_rate.clip()

    def _update_critic(self, batch: TrainingBatch):
        q_loss, rate_loss = self.compute_q_and_rate_loss(batch)

        for optimizer in [self.q1_optimizer, self.q2_optimizer, self.rate_optimizer]:
            optimizer.zero_grad()

        for loss in [q_loss, rate_loss]:
            loss.backward()

        for optimizer in [self.q1_optimizer, self.q2_optimizer, self.rate_optimizer]:
            optimizer.step()

        self.num_q_update += 1

    def compute_reset_losses(self, batch: TrainingBatch):
        with torch.no_grad():
            next_actions: torch.Tensor = self.policy_smoothing_func(
                self.policy_target(batch.next_state).sample()
            )

            current_reset_q = torch.flatten(self.reset_q_target((batch.state, batch.action)))
            next_reset_q = torch.flatten(self.reset_q_target((batch.next_state, next_actions)))
            reward = torch.isnan(batch.reward).float()

            reset_q_target = reward - self.reset_rate_target() + next_reset_q
            reset_rate_target = reward - current_reset_q + next_reset_q

        reset_rate_pred = self.reset_rate()
        reset_q_pred: torch.Tensor = torch.flatten(self.reset_q((batch.state, batch.action)))

        reset_q_loss = F.mse_loss(reset_q_pred, reset_q_target)
        reset_rate_loss = F.mse_loss(reset_rate_pred, reset_rate_target.mean())

        reset_cost = self.reset_cost()
        reset_cost_loss = -torch.mean(
            reset_cost * (float(self.reset_rate()) - self.target_terminal_probability)
        )

        if self.stats is not None:
            self.stats("reset_q").extend(reset_q_pred.detach().cpu().numpy())
            self.stats("reset_rate").append(float(reset_rate_pred))
            self.stats("reset_cost").append(float(reset_cost))
            self.stats("reset_cost_loss").append(float(reset_cost_loss))

        return reset_q_loss, reset_rate_loss, reset_cost_loss

    def compute_q_and_rate_loss(self, batch: TrainingBatch):
        with torch.no_grad():
            next_actions: torch.Tensor = self.policy_smoothing_func(
                self.policy_target(batch.next_state).sample()
            )
            next_q1 = torch.flatten(self.q1_target((batch.next_state, next_actions)))
            next_q2 = torch.flatten(self.q2_target((batch.next_state, next_actions)))
            next_q = torch.min(next_q1, next_q2)

            reward = torch.nan_to_num(batch.reward, -float(self.reset_cost()))

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
