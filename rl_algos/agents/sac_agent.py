import copy
import logging
from typing import Optional, Tuple, Union, Type, Dict, Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import cuda, distributions, nn
from torch.optim import Adam, Optimizer

import rl_algos
from rl_algos.agents.agent_base import AgentBase, AttributeSavingMixin
from rl_algos.buffers import ReplayBuffer, TrainingBatch
from rl_algos.modules import evaluating, ortho_init
from rl_algos.modules.distributions import SquashedDiagonalGaussianHead
from rl_algos.utils import synchronize_parameters
from rl_algos.utils.statistics import Statistics


class TemperatureHolder(nn.Module):
    """Module that holds a temperature as a learnable value.

    Args:
        initial_log_temperature (float): Initial value of log(temperature).
    """

    def __init__(self, initial_log_temperature: float = 0.0):
        super(TemperatureHolder, self).__init__()
        self.initial_log_temperature_value = initial_log_temperature
        self.log_temperature = nn.Parameter(
            torch.tensor(initial_log_temperature, dtype=torch.float32)
        )

    def forward(self):
        """Return a temperature as a torch.Tensor."""
        return torch.exp(self.log_temperature)

    def reset_parameters(self):
        self.log_temperature = nn.Parameter(
            torch.tensor(self.initial_log_temperature_value, dtype=torch.float32)
        )


def default_temparature_fn():
    temparature = TemperatureHolder()
    return temparature


def default_q_fn(dim_state, dim_action):
    net = nn.Sequential(
        rl_algos.modules.ConcatStateAction(),
        ortho_init(nn.Linear(dim_state + dim_action, 256), gain=np.sqrt(1.0 / 3.0)),
        nn.ReLU(),
        ortho_init(nn.Linear(256, 256), gain=np.sqrt(1.0 / 3.0)),
        nn.ReLU(),
        ortho_init(nn.Linear(256, 1), gain=np.sqrt(1.0 / 3.0)),
    )

    return net


def default_policy_fn(dim_state, dim_action):
    net = nn.Sequential(
        ortho_init(nn.Linear(dim_state, 256), gain=np.sqrt(1.0 / 3.0)),
        nn.ReLU(),
        ortho_init(nn.Linear(256, 256), gain=np.sqrt(1.0 / 3.0)),
        nn.ReLU(),
        ortho_init(nn.Linear(256, dim_action * 2), gain=np.sqrt(1.0 / 3.0)),
        SquashedDiagonalGaussianHead(),
    )

    return net


class SAC(AttributeSavingMixin, AgentBase):

    saved_attributes = (
        "q1",
        "q2",
        "q1_target",
        "q2_target",
        "policy_optimizer",
        "q_optimizer",
        "temperature_holder",
        "temperature_optimizer",
    )

    def __init__(
        self,
        dim_state: int,
        dim_action: int,
        q_fn=default_q_fn,
        policy_fn=default_policy_fn,
        temperature_fn=default_temparature_fn,
        target_entropy: Optional[float] = None,
        replay_buffer: ReplayBuffer = ReplayBuffer(1e6),
        batch_size: int = 256,
        gamma: float = 0.99,
        tau: float = 5e-3,
        replay_start_size: int = 1e4,
        optimizer_class: Type[Optimizer] = Adam,
        optimizer_kwargs: Dict[str, Any] = {"lr": 3e-4},
        device: Union[str, torch.device] = torch.device("cuda:0" if cuda.is_available() else "cpu"),
        logger=logging.getLogger(__name__),
        calc_stats: bool = True,
    ):
        self.logger: logging.Logger = logger
        if isinstance(device, str):
            device = torch.device(device)
        self.device = device

        self.dim_state = dim_state
        self.dim_action = dim_action

        # configure Q

        self.q1 = q_fn(self.dim_state, self.dim_action).to(self.device)
        self.q1_optimizer = optimizer_class(self.q1.parameters(), **optimizer_kwargs)

        self.q2 = q_fn(self.dim_state, self.dim_action).to(self.device)
        self.q2_optimizer = optimizer_class(self.q2.parameters(), **optimizer_kwargs)

        self.q1_target = copy.deepcopy(self.q1).eval().requires_grad_(False)
        self.q2_target = copy.deepcopy(self.q2).eval().requires_grad_(False)

        # configure Policy
        self.policy = policy_fn(dim_state, dim_action).to(self.device)
        self.policy_optimizer = optimizer_class(self.policy.parameters(), **optimizer_kwargs)

        self.policy_head = self.policy[-1]

        # configure Temperature
        self.temperature_holder = temperature_fn().to(self.device)
        self.temperature_optimizer = optimizer_class(
            self.temperature_holder.parameters(), **optimizer_kwargs
        )

        if target_entropy is None:
            self.target_entropy = -float(self.dim_action)
        else:
            self.target_entropy = float(target_entropy)

        # configure Replay Buffer
        self.replay_buffer = replay_buffer
        self.batch_size = batch_size

        # discount factor
        self.gamma = gamma

        # soft update parameter
        self.tau = tau

        self.replay_start_size = replay_start_size

        self.stats = Statistics() if calc_stats else None

    def act(self, state: np.ndarray) -> np.ndarray:
        state = torch.tensor(state, device=self.device, requires_grad=False)
        if self.training:
            if len(self.replay_buffer) <= self.replay_start_size:
                action = torch.rand(len(state), self.dim_action) * 2.0 - 1.0
            else:
                action_distrib: distributions.Distribution = self.policy(state)
                action: torch.Tensor = action_distrib.sample()
        else:
            with self.policy_head.deterministic():
                action = self.policy(state)

        action = action.detach().cpu().numpy()
        return action

    def observe(self, states, next_states, actions, rewards, terminals, resets):
        if self.training:
            for state, next_state, action, reward, terminal, reset in zip(
                states, next_states, actions, rewards, terminals, resets
            ):
                self.replay_buffer.append(
                    state=state,
                    next_state=next_state,
                    action=action,
                    reward=reward,
                    terminal=terminal,
                    reset=reset,
                )
            self.update_if_dataset_is_ready()

    def update_if_dataset_is_ready(self):
        assert self.training
        self.just_updated = False
        if len(self.replay_buffer) > self.replay_start_size:
            self.just_updated = True
            sampled = self.replay_buffer.sample(self.batch_size)
            batch = TrainingBatch(**sampled, device=self.device)
            self._update_q(batch)
            self._update_policy_and_temperature(batch)
            self._sync_target_network()

    def _update_q(self, batch: TrainingBatch):
        q_loss = self.compute_q_loss(batch)

        self.q1_optimizer.zero_grad()
        self.q2_optimizer.zero_grad()
        q_loss.backward()
        self.q1_optimizer.step()
        self.q2_optimizer.step()

    def _update_policy_and_temperature(self, batch):
        self.policy_loss, self.temperature_loss = self.compute_policy_and_temperature_loss(batch)

        self.policy_optimizer.zero_grad()
        self.policy_loss.backward()
        self.policy_optimizer.step()

        self.temperature_optimizer.zero_grad()
        self.temperature_loss.backward()
        self.temperature_optimizer.step()

    def _sync_target_network(self):
        """Synchronize target network with current network."""
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

    def compute_q_loss(self, batch: TrainingBatch):
        with torch.no_grad(), evaluating(self.policy, self.q1_target, self.q2_target):
            next_action_distrib: distributions.Distribution = self.policy(batch.next_state)
            next_action = next_action_distrib.sample()
            next_log_prob = next_action_distrib.log_prob(next_action)
            next_q = torch.min(
                self.q1_target((batch.next_state, next_action)),
                self.q2_target((batch.next_state, next_action)),
            )
            entropy_term = float(self.temperature_holder()) * next_log_prob[..., None]
            q_target = batch.reward + batch.terminal.logical_not() * self.gamma * torch.flatten(
                next_q - entropy_term
            )

        q1_pred = torch.flatten(self.q1((batch.state, batch.action)))
        q2_pred = torch.flatten(self.q2((batch.state, batch.action)))

        loss = 0.5 * (F.mse_loss(q1_pred, q_target) + F.mse_loss(q2_pred, q_target))

        if self.stats is not None:
            self.stats("q1_pred").extend(q1_pred.detach().cpu().numpy())
            self.stats("q2_pred").extend(q2_pred.detach().cpu().numpy())
            self.stats("q_loss").append(float(loss))

        return loss

    def compute_policy_and_temperature_loss(
        self, batch: TrainingBatch
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        action_distrib: distributions.Distribution = self.policy(batch.state)
        action = action_distrib.rsample()
        log_prob: torch.Tensor = action_distrib.log_prob(action)
        q: torch.Tensor = torch.min(self.q1((batch.state, action)), self.q2((batch.state, action)))
        policy_loss = torch.mean(float(self.temperature_holder()) * log_prob - q.flatten())

        temperature_loss = -torch.mean(
            self.temperature_holder() * (log_prob.detach() + self.target_entropy)
        )
        if self.stats is not None:
            self.stats("policy_loss").append(float(policy_loss))
            self.stats("temperature_loss").append(float(temperature_loss))
            self.stats("entropy").extend(-log_prob.detach().cpu().numpy())
        return policy_loss, temperature_loss
