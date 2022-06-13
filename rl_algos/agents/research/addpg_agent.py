import copy
import logging
from typing import Optional, Type, Union

import torch
import torch.nn.functional as F
from torch import cuda, nn
from torch.optim import Adam, Optimizer

from rl_algos.agents.agent_base import AgentBase, AttributeSavingMixin
from rl_algos.agents.td3_agent import (
    default_policy,
    default_q,
    default_target_policy_smoothing_func,
)
from rl_algos.buffers import ReplayBuffer, TrainingBatch
from rl_algos.explorers import ExplorerBase, GaussianExplorer
from rl_algos.modules import evaluating, distributions
from rl_algos.utils import synchronize_parameters
from rl_algos.utils.statistics import Statistics


class RateHolder(nn.Module):
    def __init__(self, initial_rate: float = 0.0):
        super().__init__()
        self.rate = nn.Parameter(torch.tensor(initial_rate, dtype=torch.float32))

    def forward(self):
        return self.rate


class ResetCostHolder(nn.Module):
    def __init__(self, initial_reset_cost_param: float = 0.0):
        super().__init__()
        self.reset_cost_param = nn.Parameter(
            torch.tensor(initial_reset_cost_param, dtype=torch.float32)
        )

    def forward(self):
        return F.softplus(self.reset_cost_param)


class ADDPG(AttributeSavingMixin, AgentBase):
    saved_attributes = (
        "q",
        "q_target",
        "q_optimizer",
        "policy",
        "policy_target",
        "policy_optimizer",
        "rate",
        "rate_target",
        "rate_optimizer",
        "reset_cost",
        "reset_cost_optimizer",
    )

    def __init__(
        self,
        dim_state: int,
        dim_action: int,
        optimizer_class: Type[Optimizer] = Adam,
        optimizer_kwargs: dict = {"lr": 1e-3},
        q: Optional[nn.Module] = None,
        q_optimizer_class: Type[Optimizer] = Adam,
        q_optimizer_kwargs: dict = {"lr": 3e-4},
        policy: Optional[nn.Module] = None,
        policy_optimizer_class: Type[Optimizer] = Adam,
        policy_optimizer_kwargs: dict = {"lr": 3e-4},
        policy_update_delay: int = 2,
        policy_smoothing_func=default_target_policy_smoothing_func,
        tau: float = 5e-3,
        reset_cost: Optional[float] = None,
        target_terminal_probability: float = 1 / 200,
        explorer: ExplorerBase = GaussianExplorer(0.1, -1, 1),
        replay_buffer: ReplayBuffer = ReplayBuffer(10**6),
        batch_size: int = 256,
        replay_start_size: int = 25e3,
        calc_stats: bool = True,
        logger: logging.Logger = logging.getLogger(__name__),
        device: Union[str, torch.device] = torch.device("cuda:0" if cuda.is_available() else "cpu"),
    ) -> None:
        super().__init__()
        if isinstance(device, str):
            device = torch.device(device)
        self.device = device

        self.dim_state = dim_state
        self.dim_action = dim_action

        self.policy = (
            default_policy(dim_state, dim_action).to(self.device)
            if policy is None
            else policy.to(self.device)
        )
        self.policy_optimizer = policy_optimizer_class(
            self.policy.parameters(), **policy_optimizer_kwargs
        )
        self.policy_target = copy.deepcopy(self.policy).eval().requires_grad_(False)

        self.policy_update_delay = policy_update_delay
        self.policy_smoothing_func = policy_smoothing_func

        # configure Q
        self.q = (
            default_q(dim_state, dim_action).to(self.device) if q is None else q.to(self.device)
        )
        self.q_target = copy.deepcopy(self.q).eval().requires_grad_(False)
        self.q_optimizer = q_optimizer_class(self.q.parameters(), **q_optimizer_kwargs)

        # configure Reward Rate
        self.rate = RateHolder().to(self.device)
        self.rate_target = copy.deepcopy(self.rate).eval().requires_grad_(False)
        self.rate_optimizer = optimizer_class(self.rate.parameters(), **optimizer_kwargs)

        # configure Reset Cost
        if reset_cost is not None:
            self.reset_cost = lambda: reset_cost
        else:
            self.reset_cost = ResetCostHolder().to(self.device)
            self.reset_cost_optimizer = optimizer_class(
                self.reset_cost.parameters(), **optimizer_kwargs
            )
        self.target_terminal_probability = target_terminal_probability

        self.replay_buffer = replay_buffer
        self.batch_size = batch_size
        self.replay_start_size = replay_start_size
        self.explorer = explorer

        self.t = 0
        self.tau = tau
        self.num_policy_update = 0
        self.num_q_update = 0

        self.logger = logger

        self.stats = Statistics() if calc_stats else None

    def observe(self, states, next_states, actions, rewards, terminals, resets) -> None:
        if self.training:
            for state, next_state, action, reward, terminal, reset in zip(
                states, next_states, actions, rewards, terminals, resets
            ):
                self.t += 1
                self.replay_buffer.append(
                    state=state,
                    next_state=next_state,
                    action=action,
                    reward=reward,
                    terminal=terminal,
                    reset=reset,
                )
            self.update_if_dataset_is_ready()

    def act(self, states):
        with torch.no_grad():
            states = torch.tensor(states, device=self.device)
            if self.training:
                if self.num_policy_update == 0:
                    actions = torch.rand(len(states), self.dim_action) * 2 - 1
                else:
                    action_distribs: distributions.Distribution = self.policy(states)
                    actions: torch.Tensor = action_distribs.sample()
                    actions = self.explorer.select_action(self.t, lambda: actions)
            else:
                with self.policy[-1].deterministic():
                    actions = self.policy(states)

        actions = actions.detach().cpu().numpy()
        return actions

    def update_if_dataset_is_ready(self):
        assert self.training
        self.just_updated = False
        if len(self.replay_buffer) > self.replay_start_size:
            self.just_updated = True
            if self.num_q_update == 0:
                self.logger.info("Start Update")
            batch = self.sample_batch()

            self._update_rate(batch)

            self._update_critic(batch)
            if isinstance(self.reset_cost, ResetCostHolder):
                self._update_reset_cost(batch)
            if self.num_q_update % self.policy_update_delay == 0:
                self._update_actor(batch)
                self._sync_target_network()

    def sample_batch(self):
        sampled = self.replay_buffer.sample(self.batch_size)
        return TrainingBatch(**sampled, device=self.device)

    def _update_critic(self, batch: TrainingBatch):
        q_loss = self.compute_q_loss(batch)
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()
        self.num_q_update += 1

    def _update_actor(self, batch: TrainingBatch):
        policy_loss = self.compute_policy_loss(batch)
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        self.num_policy_update += 1

    def _update_rate(self, batch: TrainingBatch):
        rate_loss = self.compute_rate_loss(batch)
        self.rate_optimizer.zero_grad()
        rate_loss.backward()
        self.rate_optimizer.step()

    def compute_rate_loss(self, batch: TrainingBatch):
        with torch.no_grad(), evaluating(self.q_target):
            next_actions: torch.Tensor = self.policy_target(batch.next_state).sample()
            next_q = torch.flatten(self.q_target((batch.next_state, next_actions)))
            current_q = torch.flatten(self.q_target((batch.state, batch.action)))
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
            next_actions: torch.Tensor = self.policy_smoothing_func(
                self.policy_target(batch.next_state).sample()
            )
            next_q = torch.flatten(self.q_target((batch.next_state, next_actions)))
            q_target: torch.Tensor = (
                torch.nan_to_num(batch.reward, -float(self.reset_cost()))
                - self.rate_target()
                + next_q
            )
        q_pred = torch.flatten(self.q((batch.state, batch.action)))

        loss = F.mse_loss(q_pred, q_target)

        if self.stats is not None:
            self.stats("q_pred").extend(q_pred.detach().cpu().numpy())
            self.stats("q_target").extend(q_target.detach().cpu().numpy())
            self.stats("q_loss").append(loss.detach().cpu().numpy())

        return loss

    def compute_policy_loss(self, batch: TrainingBatch):
        actions = self.policy(batch.state).rsample()
        q = self.q((batch.state, actions))
        policy_loss = -torch.mean(q)
        if self.stats is not None:
            self.stats("policy_loss").append(float(policy_loss))
        return policy_loss

    def _sync_target_network(self):
        """Synchronize target network with current network."""
        synchronize_parameters(
            src=self.policy,
            dst=self.policy_target,
            method="soft",
            tau=self.tau,
        )
        synchronize_parameters(
            src=self.q,
            dst=self.q_target,
            method="soft",
            tau=self.tau,
        )
        synchronize_parameters(src=self.rate, dst=self.rate_target, method="soft", tau=self.tau)
