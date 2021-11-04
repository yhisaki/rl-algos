import collections
import copy
import logging
from typing import Any, Optional, Tuple, Type, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import cuda, distributions, nn
from torch.optim import Adam, Optimizer

import rlrl
from rlrl.agents.agent_base import AgentBase, AttributeSavingMixin
from rlrl.modules import evaluating, ortho_init
from rlrl.modules.distributions import SquashedDiagonalGaussianHead, StochasticHeadBase
from rlrl.replay_buffers import ReplayBuffer, TorchTensorBatch
from rlrl.utils import clear_if_maxlen_is_none, mean_or_nan, synchronize_parameters


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


class SacAgent(AttributeSavingMixin, AgentBase):

    saved_attributes = (
        "q1",
        "q2",
        "q1_target",
        "q2_target",
        "policy_optimizer",
        "q1_optimizer",
        "q2_optimizer",
        "temperature_holder",
        "temperature_optimizer",
    )

    def __init__(
        self,
        dim_state: int,
        dim_action: int,
        q1: Optional[nn.Module] = None,
        q2: Optional[nn.Module] = None,
        q_optimizer_class: Type[Optimizer] = Adam,
        q_optimizer_kwargs: dict = {"lr": 1e-3},
        policy: Optional[nn.Module] = None,
        policy_optimizer_class: Type[Optimizer] = Adam,
        policy_optimizer_kwargs: dict = {"lr": 1e-3},
        temperature_holder: nn.Module = TemperatureHolder(),
        temperature_optimizer_class: Type[Optimizer] = Adam,
        temperature_optimizer_kwargs: dict = {"lr": 1e-3},
        target_entropy: Optional[float] = None,
        replay_buffer: ReplayBuffer = ReplayBuffer(1e6),
        batch_size: int = 256,
        gamma: float = 0.99,
        tau_q: float = 5e-3,
        num_random_act: int = 1e4,
        device: Union[str, torch.device] = torch.device("cuda:0" if cuda.is_available() else "cpu"),
        logger=logging.getLogger(__name__),
        calc_stats: bool = True,
        q_stats_window=None,
        q_loss_stats_window=None,
        policy_loss_stats_window=None,
        entropy_stats_window=None,
        temperature_loss_stats_window=None,
    ):
        self.logger: logging.Logger = logger
        if isinstance(device, str):
            device = torch.device(device)
        self.device = device

        self.dim_state = dim_state
        self.dim_action = dim_action

        # configure Q
        if q1 is None:
            self.q1 = nn.Sequential(
                rlrl.modules.ConcatStateAction(),
                ortho_init(
                    nn.Linear(self.dim_state + self.dim_action, 256), gain=np.sqrt(1.0 / 3.0)
                ),
                nn.ReLU(),
                ortho_init(nn.Linear(256, 256), gain=np.sqrt(1.0 / 3.0)),
                nn.ReLU(),
                ortho_init(nn.Linear(256, 1), gain=np.sqrt(1.0 / 3.0)),
            ).to(self.device)
        else:
            self.q1 = q1.to(self.device)
        if q2 is None:
            self.q2 = nn.Sequential(
                rlrl.modules.ConcatStateAction(),
                ortho_init(
                    nn.Linear(self.dim_state + self.dim_action, 256), gain=np.sqrt(1.0 / 3.0)
                ),
                nn.ReLU(),
                ortho_init(nn.Linear(256, 256), gain=np.sqrt(1.0 / 3.0)),
                nn.ReLU(),
                ortho_init(nn.Linear(256, 1), gain=np.sqrt(1.0 / 3.0)),
            ).to(self.device)
        else:
            self.q2 = q2.to(self.device)

        self.q1_target = copy.deepcopy(self.q1).eval().requires_grad_(False)
        self.q2_target = copy.deepcopy(self.q1).eval().requires_grad_(False)

        self.q1_optimizer = q_optimizer_class(self.q1.parameters(), **q_optimizer_kwargs)
        self.q2_optimizer = q_optimizer_class(self.q2.parameters(), **q_optimizer_kwargs)

        # configure Policy
        if policy is None:
            self.policy = nn.Sequential(
                ortho_init(nn.Linear(self.dim_state, 256), gain=np.sqrt(1.0 / 3.0)),
                nn.ReLU(),
                ortho_init(nn.Linear(256, 256), gain=np.sqrt(1.0 / 3.0)),
                nn.ReLU(),
                ortho_init(nn.Linear(256, self.dim_action * 2), gain=np.sqrt(1.0 / 3.0)),
                SquashedDiagonalGaussianHead(),
            ).to(self.device)
        else:
            self.policy = policy.to(self.device)

        self.policy_head: StochasticHeadBase = self.policy[-1]

        if not isinstance(self.policy_head, StochasticHeadBase):
            self.logger.warning("policy head is not stochasitic!!")

        self.policy_optimizer = policy_optimizer_class(
            self.policy.parameters(), **policy_optimizer_kwargs
        )

        # configure Temperature
        self.temperature_holder = temperature_holder.to(self.device)
        self.temperature_optimizer = temperature_optimizer_class(
            self.temperature_holder.parameters(), **temperature_optimizer_kwargs
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
        if not ((0.0 < self.gamma) & (self.gamma < 1.0)):
            raise ValueError("The discount rate must be greater than zero and less than one.")

        # soft update parameter
        self.tau_q = tau_q

        self.num_random_act = num_random_act

        self.calc_stats = calc_stats

        if self.calc_stats:
            self.q1_record = collections.deque(maxlen=q_stats_window)
            self.q2_record = collections.deque(maxlen=q_stats_window)
            self.q1_loss_record = collections.deque(maxlen=q_loss_stats_window)
            self.q2_loss_record = collections.deque(maxlen=q_loss_stats_window)
            self.policy_loss_record = collections.deque(maxlen=policy_loss_stats_window)
            self.entropy_record = collections.deque(maxlen=entropy_stats_window)
            self.temperature_loss_record = collections.deque(maxlen=temperature_loss_stats_window)

    def act(self, state: np.ndarray) -> np.ndarray:
        state = torch.tensor(state, device=self.device, requires_grad=False)
        if self.training:
            if len(self.replay_buffer) <= self.num_random_act:
                action = torch.rand(len(state), self.dim_action) * 2.0 - 1.0
            else:
                action_distrib: distributions.Distribution = self.policy(state)
                action: torch.Tensor = action_distrib.sample()
        else:
            with self.policy_head.deterministic():
                action = self.policy(state)

        action = action.detach().cpu().numpy()
        return action

    def observe(self, states, next_states, actions, rewards, terminals):
        if self.training:
            for state, next_state, action, reward, terminal in zip(
                states, next_states, actions, rewards, terminals
            ):
                self.replay_buffer.append(
                    state=state,
                    next_state=next_state,
                    action=action,
                    reward=reward,
                    terminal=terminal,
                )
            self.update_if_dataset_is_ready()

    def update_if_dataset_is_ready(self):
        assert self.training
        self.just_updated = False
        if len(self.replay_buffer) > self.num_random_act:
            self.just_updated = True
            sampled = self.replay_buffer.sample(self.batch_size)
            self.batch = TorchTensorBatch(**sampled, device=self.device)
            self._update_q(self.batch)
            self._update_policy_and_temperature(self.batch)
            self._sync_target_network()

    def _update_q(self, batch: TorchTensorBatch):
        self.q1_loss, self.q2_loss = self.compute_q_loss(batch)

        self.q1_optimizer.zero_grad()
        self.q1_loss.backward()
        self.q1_optimizer.step()

        self.q2_optimizer.zero_grad()
        self.q2_loss.backward()
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
            tau=self.tau_q,
        )
        synchronize_parameters(
            src=self.q2,
            dst=self.q2_target,
            method="soft",
            tau=self.tau_q,
        )

    def compute_q_loss(
        self, batch: Union[TorchTensorBatch, Any]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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

        predicted_q1 = torch.flatten(self.q1((batch.state, batch.action)))
        predicted_q2 = torch.flatten(self.q2((batch.state, batch.action)))

        q1_loss = 0.5 * F.mse_loss(predicted_q1, q_target)
        q2_loss = 0.5 * F.mse_loss(predicted_q2, q_target)

        if self.calc_stats:
            self.q1_record.extend(predicted_q1.detach().cpu().numpy())
            self.q2_record.extend(predicted_q2.detach().cpu().numpy())
            self.q1_loss_record.append(float(self.q1_loss))
            self.q2_loss_record.append(float(self.q2_loss))

        return q1_loss, q2_loss

    def compute_policy_and_temperature_loss(
        self, batch: TorchTensorBatch
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        action_distrib: distributions.Distribution = self.policy(batch.state)
        action = action_distrib.rsample()
        log_prob: torch.Tensor = action_distrib.log_prob(action)
        q: torch.Tensor = torch.min(self.q1((batch.state, action)), self.q2((batch.state, action)))
        policy_loss = torch.mean(float(self.temperature_holder()) * log_prob - q.flatten())

        temperature_loss = -torch.mean(
            self.temperature_holder() * (log_prob.detach() + self.target_entropy)
        )
        if self.calc_stats:
            self.policy_loss_record.append(float(policy_loss))
            self.temperature_loss_record.append(float(temperature_loss))
            self.entropy_record.extend(-log_prob.detach().cpu().numpy())
        return policy_loss, temperature_loss

    def get_statistics(self):
        if self.calc_stats:
            stats = {
                "average_q1": mean_or_nan(self.q1_record),
                "average_q2": mean_or_nan(self.q2_record),
                "average_q1_loss": mean_or_nan(self.q1_loss_record),
                "average_q2_loss": mean_or_nan(self.q2_loss_record),
                "average_policy_loss": mean_or_nan(self.policy_loss_record),
                "average_entropy": mean_or_nan(self.entropy_record),
                "average_temperature_loss": mean_or_nan(self.temperature_loss_record),
            }

            clear_if_maxlen_is_none(
                self.q1_record,
                self.q2_record,
                self.q1_loss_record,
                self.q2_loss_record,
                self.policy_loss_record,
                self.entropy_record,
                self.temperature_loss_record,
            )

            return stats
        else:
            self.logger.warning("get_statistics() is called even though the calc_stats is False.")
