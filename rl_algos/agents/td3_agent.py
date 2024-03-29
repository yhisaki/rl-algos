import copy
import logging
from typing import Any, Dict, Type, Union

import torch
import torch.nn.functional as F
from torch import cuda, distributions, nn
from torch.optim import Adam, Optimizer

from rl_algos.agents.agent_base import AgentBase, AttributeSavingMixin
from rl_algos.buffers import ReplayBuffer, TrainingBatch
from rl_algos.explorers import ExplorerBase, GaussianExplorer
from rl_algos.modules import ConcatStateAction, evaluating
from rl_algos.modules.distributions import DeterministicHead
from rl_algos.utils import logger, synchronize_parameters
from rl_algos.utils.statistics import Statistics


def default_target_policy_smoothing_func(batch_action):
    """Add noises to actions for target policy smoothing."""
    noise = torch.clamp(0.2 * torch.randn_like(batch_action), -0.5, 0.5)
    return torch.clamp(batch_action + noise, -1, 1)


def default_policy_fn(dim_state, dim_action):
    net = nn.Sequential(
        nn.Linear(dim_state, 400),
        nn.ReLU(),
        nn.Linear(400, 300),
        nn.ReLU(),
        nn.Linear(300, dim_action),
        nn.Tanh(),
        DeterministicHead(),
    )

    return net


def default_q_fn(dim_state, dim_action):
    net = nn.Sequential(
        ConcatStateAction(),
        nn.Linear(dim_state + dim_action, 400),
        nn.ReLU(),
        nn.Linear(400, 300),
        nn.ReLU(),
        nn.Linear(300, 1),
    )

    return net


class TD3(AttributeSavingMixin, AgentBase):
    saved_attributes = (
        "q1",
        "q2",
        "q1_target",
        "q2_target",
        "q1_optimizer",
        "q2_optimizer",
        "policy",
        "policy_target",
        "policy_optimizer",
        "replay_buffer",
    )

    def __init__(
        self,
        dim_state: int,
        dim_action: int,
        q_fn=default_q_fn,
        policy_fn=default_policy_fn,
        policy_update_delay: int = 2,
        policy_smoothing_func=default_target_policy_smoothing_func,
        tau: float = 5e-3,
        explorer: ExplorerBase = GaussianExplorer(0.1, -1, 1),
        gamma: float = 0.99,
        replay_buffer: ReplayBuffer = ReplayBuffer(10**6),
        batch_size: int = 256,
        replay_start_size: int = 25e3,
        calc_stats: bool = True,
        optimizer_class: Type[Optimizer] = Adam,
        optimizer_kwargs: Dict[str, Any] = {"lr": 1e-3},
        logger: logging.Logger = logger,
        device: Union[str, torch.device] = torch.device("cuda:0" if cuda.is_available() else "cpu"),
    ) -> None:
        super().__init__()
        if isinstance(device, str):
            device = torch.device(device)
        self.device = device

        self.dim_state = dim_state
        self.dim_action = dim_action
        self.gamma = gamma

        self.policy = policy_fn(self.dim_state, self.dim_action).to(self.device)
        self.policy_optimizer = optimizer_class(self.policy.parameters(), **optimizer_kwargs)
        self.policy_target = copy.deepcopy(self.policy).eval().requires_grad_(False)

        self.policy_update_delay = policy_update_delay
        self.policy_smoothing_func = policy_smoothing_func

        # configure Q
        self.q1 = q_fn(self.dim_state, self.dim_action).to(self.device)
        self.q1_optimizer = optimizer_class(self.q1.parameters(), **optimizer_kwargs)

        self.q2 = q_fn(self.dim_state, self.dim_action).to(self.device)
        self.q2_optimizer = optimizer_class(self.q2.parameters(), **optimizer_kwargs)

        self.q1_target = copy.deepcopy(self.q1).eval().requires_grad_(False)
        self.q2_target = copy.deepcopy(self.q2).eval().requires_grad_(False)

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
        q_loss = self.compute_q_loss(batch)
        self.q1_optimizer.zero_grad()
        self.q2_optimizer.zero_grad()
        q_loss.backward()
        self.q1_optimizer.step()
        self.q2_optimizer.step()
        self.num_q_update += 1

    def _update_actor(self, batch: TrainingBatch):
        policy_loss = self.compute_policy_loss(batch)

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        self.num_policy_update += 1

    def compute_q_loss(self, batch: TrainingBatch):
        with torch.no_grad(), evaluating(self.policy_target, self.q1_target, self.q2_target):
            next_actions: torch.Tensor = self.policy_smoothing_func(
                self.policy_target(batch.next_state).sample()
            )
            next_q1 = self.q1_target((batch.next_state, next_actions))
            next_q2 = self.q2_target((batch.next_state, next_actions))
            next_q = torch.min(next_q1, next_q2)

            q_target = batch.reward + self.gamma * batch.terminal.logical_not() * torch.flatten(
                next_q
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

    def compute_policy_loss(self, batch: TrainingBatch):
        actions = self.policy(batch.state).rsample()
        q = self.q1((batch.state, actions))
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
            src=self.q1,
            dst=self.q1_target,
            method="soft",
            tau=self.tau,
        )
        synchronize_parameters(
            src=self.q2,
            dst=self.q2_target,
            method="soft",
            tau=self.tau,
        )
