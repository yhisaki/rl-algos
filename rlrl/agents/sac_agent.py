import copy
import logging
from typing import Any, Optional, Tuple, Type, Union

import numpy as np
import rlrl
import torch
import torch.nn.functional as F
from rlrl.agents.agent_base import AgentBase, AttributeSavingMixin
from rlrl.modules import evaluating
from rlrl.modules.distributions import SquashedDiagonalGaussianHead, StochanicHeadBase
from rlrl.replay_buffers import ReplayBuffer, TorchTensorBatch
from rlrl.utils import synchronize_parameters
from torch import cuda, distributions, nn
from torch.optim import Adam, Optimizer


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
        tau: float = 5e-3,
        num_random_act: int = 1e4,
        device: Union[str, torch.device] = torch.device("cuda:0" if cuda.is_available() else "cpu"),
        logger=logging.getLogger(__name__),
        **kwargs,
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
                nn.Linear(self.dim_state + self.dim_action, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, 1),
            ).to(self.device)
            self.q1.apply(SacAgent.default_network_initializer)
        else:
            self.q1 = q1.to(self.device)
        if q2 is None:
            self.q2 = nn.Sequential(
                rlrl.modules.ConcatStateAction(),
                nn.Linear(self.dim_state + self.dim_action, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, 1),
            ).to(self.device)
            self.q2.apply(SacAgent.default_network_initializer)
        else:
            self.q2 = q2.to(self.device)

        self.q1_target = copy.deepcopy(self.q1).eval().requires_grad_(False)
        self.q2_target = copy.deepcopy(self.q1).eval().requires_grad_(False)

        self.q1_optimizer = q_optimizer_class(self.q1.parameters(), **q_optimizer_kwargs)
        self.q2_optimizer = q_optimizer_class(self.q2.parameters(), **q_optimizer_kwargs)

        # configure Policy
        if policy is None:
            self.policy = nn.Sequential(
                nn.Linear(self.dim_state, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, self.dim_action * 2),
                SquashedDiagonalGaussianHead(),
            ).to(self.device)
            self.policy.apply(SacAgent.default_network_initializer)
        else:
            self.policy = policy.to(self.device)

        self.policy_head: StochanicHeadBase = self.policy[-1]

        if not isinstance(self.policy_head, StochanicHeadBase):
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
        self.tau = tau

        self.num_random_act = num_random_act

    def act(
        self,
        state: np.ndarray,
        compute_entropy: bool = False,
    ) -> np.ndarray:
        state = torch.tensor(state, device=self.device, requires_grad=False)
        if self.training:
            if len(self.replay_buffer) <= self.num_random_act:
                action = torch.rand(len(state), self.dim_action)
            else:
                action_distrib: distributions.Distribution = self.policy(state)
                action: torch.Tensor = action_distrib.sample()
                if compute_entropy:
                    self.entropy = action_distrib.log_prob(action).detach().cpu()
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
        if len(self.replay_buffer) > self.num_random_act:
            sampled = self.replay_buffer.sample(self.batch_size)
            self.batch = TorchTensorBatch(**sampled, device=self.device)
            self._update_q(self.batch)
            self._update_policy_and_temperature(self.batch)
            self._sync_target_network()

    def _update_q(self, batch: TorchTensorBatch):
        self.q1_loss, self.q2_loss = SacAgent.compute_q_loss(
            batch=batch,
            policy=self.policy,
            temperature=float(self.temperature_holder()),
            gamma=self.gamma,
            q1=self.q1,
            q2=self.q2,
            q1_target=self.q1_target,
            q2_target=self.q2_target,
        )
        self.q1_optimizer.zero_grad()
        self.q1_loss.backward()
        self.q1_optimizer.step()

        self.q2_optimizer.zero_grad()
        self.q2_loss.backward()
        self.q2_optimizer.step()

    def _update_policy_and_temperature(self, batch):
        self.policy_loss, self.temperature_loss = SacAgent.compute_policy_and_temperature_loss(
            batch, self.policy, self.temperature_holder, self.q1, self.q2, self.target_entropy
        )

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

    @staticmethod
    def configure_agent_from_gym(env, **kwargs):
        """state space must be a 1d vector"""
        return SacAgent(
            dim_state=env.observation_space.shape[0],
            dim_action=env.action_space.shape[0],
            **kwargs,
        )

    @staticmethod
    @torch.no_grad()
    def default_network_initializer(layer):
        if isinstance(layer, nn.Linear):
            nn.init.orthogonal_(layer.weight, gain=np.sqrt(1.0 / 3.0))
            nn.init.zeros_(layer.bias)

    @staticmethod
    def compute_q_loss(
        batch: Union[TorchTensorBatch, Any],
        policy: nn.Module,
        temperature: float,
        gamma: float,
        q1: nn.Module,
        q2: nn.Module,
        q1_target: nn.Module,
        q2_target: nn.Module,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad(), evaluating(policy, q1_target, q2_target):
            next_action_distrib: distributions.Distribution = policy(batch.next_state)
            next_action = next_action_distrib.sample()
            next_log_prob = next_action_distrib.log_prob(next_action)
            next_q = torch.min(
                q1_target((batch.next_state, next_action)),
                q2_target((batch.next_state, next_action)),
            )
            entropy_term = temperature * next_log_prob[..., None]
            q_target = batch.reward + batch.terminal.logical_not() * gamma * torch.flatten(
                next_q - entropy_term
            )

        predicted_q1 = torch.flatten(q1((batch.state, batch.action)))
        predicted_q2 = torch.flatten(q2((batch.state, batch.action)))

        q1_loss = 0.5 * F.mse_loss(predicted_q1, q_target)
        q2_loss = 0.5 * F.mse_loss(predicted_q2, q_target)
        return q1_loss, q2_loss

    @staticmethod
    def compute_policy_and_temperature_loss(
        batch: TorchTensorBatch,
        policy: nn.Module,
        temperature_holder: TemperatureHolder,
        q1: nn.Module,
        q2: nn.Module,
        entropy_target: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        action_distrib: distributions.Distribution = policy(batch.state)
        action = action_distrib.rsample()
        log_prob: torch.Tensor = action_distrib.log_prob(action)
        q: torch.Tensor = torch.min(q1((batch.state, action)), q2((batch.state, action)))
        policy_loss = (float(temperature_holder()) * log_prob - q.flatten()).mean()

        temperature_loss = SacAgent.compute_temperature_loss(
            log_prob.detach(), temperature_holder, entropy_target
        )
        return policy_loss, temperature_loss

    @staticmethod
    def compute_temperature_loss(
        log_prob: torch.Tensor, temperature: TemperatureHolder, target_entropy: float
    ) -> torch.Tensor:
        loss = -(temperature() * (log_prob + target_entropy)).mean()
        return loss
