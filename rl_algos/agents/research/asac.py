import copy
import logging
from typing import Any, Dict, Type, Union

import torch
import torch.nn.functional as F
from torch import cuda, distributions, nn
from torch.optim import Adam, Optimizer
import rl_algos

from rl_algos.agents.research.rate import RateHolder, default_rate_fn  # noqa
from rl_algos.agents.research.reset_cost import default_reset_cost_fn
from rl_algos.agents.sac_agent import SAC, default_policy_fn, default_q_fn, default_temparature_fn
from rl_algos.buffers import ReplayBuffer, TrainingBatch
from rl_algos.modules import evaluating
from rl_algos.utils.sync_param import synchronize_parameters


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


class ASAC(SAC):
    def __init__(
        self,
        dim_state: int,
        dim_action: int,
        q_fn=default_q_fn,
        policy_fn=default_policy_fn,
        temperature_fn=default_temparature_fn,
        rate_fn=default_rate_fn,
        tau: float = 5e-3,
        reset_cost_fn=default_reset_cost_fn,
        target_terminal_probability: float = 1 / 1000,
        replay_buffer: ReplayBuffer = ReplayBuffer(10**6),
        batch_size: int = 256,
        replay_start_size: int = 1e4,
        optimizer_class: Type[Optimizer] = Adam,
        optimizer_kwargs: Dict[str, Any] = {"lr": 3e-4},
        calc_stats: bool = True,
        logger: logging.Logger = logging.getLogger(__name__),
        device: Union[str, torch.device] = torch.device("cuda:0" if cuda.is_available() else "cpu"),
    ) -> None:
        super().__init__(
            dim_state=dim_state,
            dim_action=dim_action,
            q_fn=q_fn,
            policy_fn=policy_fn,
            temperature_fn=temperature_fn,
            tau=tau,
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
            sampled = self.replay_buffer.sample(self.batch_size)
            batch = TrainingBatch(**sampled, device=self.device)
            self._update_q_and_rate(batch)
            self._update_reset_cost(batch)
            self._update_policy_and_temperature(batch)
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

    def _update_q_and_rate(self, batch: TrainingBatch):
        q_loss, rate_loss = self.compute_q_and_rate_loss(batch)

        for optimizer in [self.q1_optimizer, self.q2_optimizer, self.rate_optimizer]:
            optimizer.zero_grad()

        for loss in [q_loss, rate_loss]:
            loss.backward()

        for optimizer in [self.q1_optimizer, self.q2_optimizer, self.rate_optimizer]:
            optimizer.step()

    def compute_reset_losses(self, batch: TrainingBatch):
        with torch.no_grad():
            next_action_distrib: distributions.Distribution = self.policy(batch.next_state)
            next_action = next_action_distrib.sample()

            current_reset_q = torch.flatten(self.reset_q_target((batch.state, batch.action)))
            next_reset_q = torch.flatten(self.reset_q_target((batch.next_state, next_action)))
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
        with torch.no_grad(), evaluating(self.policy, self.q1_target, self.q2_target):
            next_action_distrib: distributions.Distribution = self.policy(batch.next_state)
            next_action = next_action_distrib.sample()
            next_log_prob = next_action_distrib.log_prob(next_action)

            next_q1 = torch.flatten(self.q1_target((batch.next_state, next_action)))
            next_q2 = torch.flatten(self.q2_target((batch.next_state, next_action)))
            next_q = torch.min(next_q1, next_q2)
            entropy_term = float(self.temperature_holder()) * next_log_prob

            reward = torch.nan_to_num(batch.reward, -float(self.reset_cost()))

            current_q1 = torch.flatten(self.q1_target((batch.state, batch.action)))
            current_q2 = torch.flatten(self.q2_target((batch.state, batch.action)))

            current_q = (current_q1 + current_q2) / 2.0

            q_target: torch.Tensor = reward - self.rate_target() + (next_q - entropy_term)
            rate_target = reward - current_q + (next_q - entropy_term)

        q1_pred = torch.flatten(self.q1((batch.state, batch.action)))
        q2_pred = torch.flatten(self.q2((batch.state, batch.action)))

        q_loss = 0.5 * (F.mse_loss(q1_pred, q_target) + F.mse_loss(q2_pred, q_target))

        rate_pred = self.rate()
        rate_loss = F.mse_loss(rate_pred, rate_target.mean())

        if self.stats is not None:
            self.stats("q1_pred").extend(q1_pred.detach().cpu().numpy())
            self.stats("q2_pred").extend(q2_pred.detach().cpu().numpy())
            self.stats("q_target").extend(q_target.detach().cpu().numpy())
            self.stats("q_loss").append(float(q_loss))
            self.stats("rate_pred").append(float(rate_pred))
            self.stats("rate_targets", methods=["mean", "var"]).extend(
                rate_target.detach().cpu().numpy()
            )
        return q_loss, rate_loss

    def _sync_target_network(self):
        synchronize_parameters(
            src=self.reset_rate,
            dst=self.reset_rate_target,
            method="soft",
            tau=self.tau,
        )
        synchronize_parameters(
            src=self.reset_q,
            dst=self.reset_q_target,
            method="soft",
            tau=self.tau,
        )
        synchronize_parameters(
            src=self.rate,
            dst=self.rate_target,
            method="soft",
            tau=self.tau,
        )
        return super()._sync_target_network()
