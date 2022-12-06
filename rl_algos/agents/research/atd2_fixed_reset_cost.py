import copy
import logging
from typing import Any, Dict, Type, Union

import torch
import torch.nn.functional as F
from torch import cuda, distributions
from torch.optim import Adam, Optimizer

from rl_algos.agents.agent_base import AgentBase, AttributeSavingMixin
from rl_algos.agents.td3_agent import (
    default_policy_fn,
    default_q_fn,
    default_target_policy_smoothing_func,
)
from rl_algos.buffers import ReplayBuffer, TrainingBatch
from rl_algos.explorers import ExplorerBase, GaussianExplorer
from rl_algos.modules import evaluating
from rl_algos.utils import synchronize_parameters
from rl_algos.utils.statistics import Statistics

from .rate import default_rate_fn


class ATD2FixedResetCost(AttributeSavingMixin, AgentBase):
    saved_attributes = (
        "q",
        "q_target",
        "q_optimizer",
        "policy",
        "policy_target",
        "policy_optimizer",
        "replay_buffer",
        "rate",
        "rate_optimizer",
        "rate_target",
    )

    def __init__(
        self,
        dim_state: int,
        dim_action: int,
        rate_fn=default_rate_fn,
        q_fn=default_q_fn,
        policy_fn=default_policy_fn,
        policy_update_delay: int = 2,
        policy_smoothing_func=default_target_policy_smoothing_func,
        reset_cost=100,
        tau: float = 5e-3,
        explorer: ExplorerBase = GaussianExplorer(0.1, -1, 1),
        replay_buffer: ReplayBuffer = ReplayBuffer(10**6),
        batch_size: int = 256,
        replay_start_size: int = 25e3,
        calc_stats: bool = True,
        optimizer_class: Type[Optimizer] = Adam,
        optimizer_kwargs: Dict[str, Any] = {"lr": 1e-3},
        logger: logging.Logger = logging.getLogger(__name__),
        device: Union[str, torch.device] = torch.device("cuda:0" if cuda.is_available() else "cpu"),
    ) -> None:
        super().__init__()
        if isinstance(device, str):
            device = torch.device(device)
        self.device = device

        self.dim_state = dim_state
        self.dim_action = dim_action

        self.policy = policy_fn(self.dim_state, self.dim_action).to(self.device)
        self.policy_optimizer = optimizer_class(self.policy.parameters(), **optimizer_kwargs)
        self.policy_target = copy.deepcopy(self.policy).eval().requires_grad_(False)

        self.policy_update_delay = policy_update_delay
        self.policy_smoothing_func = policy_smoothing_func

        # configure Q
        self.q = q_fn(self.dim_state, self.dim_action).to(self.device)
        self.q_optimizer = optimizer_class(self.q.parameters(), **optimizer_kwargs)
        self.q_target = copy.deepcopy(self.q).eval().requires_grad_(False)

        self.rate = rate_fn().to(self.device)
        self.rate_optimizer = optimizer_class(self.rate.parameters(), **optimizer_kwargs)
        self.rate_target = copy.deepcopy(self.rate).eval().requires_grad_(False)

        self.reset_cost = reset_cost

        self.replay_buffer = replay_buffer
        self.batch_size = batch_size
        self.replay_start_size = replay_start_size
        self.explorer = explorer

        self.t = 0
        self.tau = tau
        self.num_policy_update = 0
        self.num_q_update = 0

        self.stats = Statistics() if calc_stats else None

        self.logger = logger

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
            sampled = self.replay_buffer.sample(self.batch_size)
            batch = TrainingBatch(**sampled, device=self.device)
            self._update_critic(batch)
            if self.num_q_update % self.policy_update_delay == 0:
                self._update_actor(batch)
                self._sync_target_network()

    def _update_critic(self, batch: TrainingBatch):
        q_loss, rate_loss = self.compute_q_and_rate_loss(batch)

        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        self.rate_optimizer.zero_grad()
        rate_loss.backward()
        self.rate_optimizer.step()

        self.num_q_update += 1

    def _update_actor(self, batch: TrainingBatch):
        policy_loss = self.compute_policy_loss(batch)

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        self.num_policy_update += 1

    def compute_q_and_rate_loss(self, batch: TrainingBatch):
        with torch.no_grad(), evaluating(self.policy_target, self.q_target):
            next_actions: torch.Tensor = self.policy_smoothing_func(
                self.policy_target(batch.next_state).sample()
            )
            next_q = torch.flatten(self.q_target((batch.next_state, next_actions)))

            reward = torch.nan_to_num(batch.reward, -self.reset_cost)

            q_target = reward - self.rate_target() + next_q
            current_q = torch.flatten(self.q_target((batch.state, batch.action)))

            rate_target = reward - current_q + next_q

        q_pred = torch.flatten(self.q((batch.state, batch.action)))
        q_loss = F.mse_loss(q_pred, q_target)

        rate_pred = self.rate()
        rate_loss = F.mse_loss(rate_pred, rate_target.mean())

        if self.stats is not None:
            self.stats("q_pred").extend(q_pred.detach().cpu().numpy())
            self.stats("q_target").extend(q_target.detach().cpu().numpy())
            self.stats("q_loss").append(q_loss.detach().cpu().numpy())
            self.stats("rate_pred").append(float(rate_pred))
            self.stats("rate_targets", methods=["mean", "var"]).extend(
                rate_target.detach().cpu().numpy()
            )

        return q_loss, rate_loss

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
        synchronize_parameters(
            src=self.rate,
            dst=self.rate_target,
            method="soft",
            tau=self.tau,
        )
