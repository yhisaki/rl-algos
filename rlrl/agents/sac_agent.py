import copy
from typing import Any, Optional, Tuple, Type, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import cuda, distributions, nn
from torch.optim import Adam, Optimizer

import rlrl
from rlrl.agents.agent_base import AgentBase, AttributeSavingMixin
from rlrl.nn import StochanicHeadBase, evaluating
from rlrl.replay_buffers import ReplayBuffer, TorchTensorBatch
from rlrl.utils import synchronize_parameters


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


class SquashedDiagonalGaussianHead(StochanicHeadBase):
    def __init__(self):
        super().__init__()

    def forward_stochanic(self, x):
        mean, log_scale = torch.chunk(x, 2, dim=x.dim() // 2)
        log_scale = torch.clamp(log_scale, -20.0, 2.0)
        var = torch.exp(log_scale * 2)
        base_distribution = distributions.Independent(
            distributions.Normal(loc=mean, scale=torch.sqrt(var)), 1
        )
        # cache_size=1 is required for numerical stability
        # https://pytorch.org/docs/stable/distributions.html#torch.distributions.transformed_distribution.TransformedDistribution
        return distributions.transformed_distribution.TransformedDistribution(
            base_distribution,
            [
                distributions.transforms.TanhTransform(cache_size=1),
            ],
        )

    def forward_determistic(self, x):
        mean, _ = torch.chunk(x, 2, dim=x.dim() // 2)
        return self.loc + self.scale * torch.tanh(mean)


class SacAgent(AgentBase, AttributeSavingMixin):

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
        device: Union[str, torch.device] = torch.device("cuda:0" if cuda.is_available() else "cpu"),
        **kwargs,
    ):
        if isinstance(device, str):
            device = torch.device(device)
        self.device = device

        self.dim_state = dim_state
        self.dim_action = dim_action

        # configure Q
        if q1 is None:
            self.q1 = nn.Sequential(
                rlrl.nn.ConcatStateAction(),
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
                rlrl.nn.ConcatStateAction(),
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

    def act(
        self,
        state: np.ndarray,
        compute_entropy: bool = False,
    ) -> np.ndarray:
        state = torch.tensor(state).to(self.device).requires_grad_(False)
        _action: Union[torch.Tensor, distributions.Distribution] = self.policy(state)
        if isinstance(_action, distributions.Distribution):
            action: torch.Tensor = _action.sample()
            if compute_entropy:
                self.entropy = float(_action.log_prob(action))
        elif isinstance(_action, torch.Tensor):
            action = _action
        action = action.detach().cpu().numpy()
        return action

    def observe(self, state, next_state, action, reward, terminal):
        self.replay_buffer.append(
            state=state,
            next_state=next_state,
            action=action,
            reward=reward,
            terminal=terminal,
        )

    def update(self):
        sampled = self.replay_buffer.sample(self.batch_size)
        self.batch = TorchTensorBatch(**sampled, device=self.device)
        self.update_q(self.batch)
        self.update_policy_and_temperature(self.batch)
        self.sync_target_network()

    def update_q(self, batch: TorchTensorBatch):
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

    def update_policy_and_temperature(self, batch):
        self.policy_loss, self.temperature_loss = SacAgent.compute_policy_and_temperature_loss(
            batch, self.policy, self.temperature_holder, self.q1, self.q2, self.target_entropy
        )

        self.policy_optimizer.zero_grad()
        self.policy_loss.backward()
        self.policy_optimizer.step()

        self.temperature_optimizer.zero_grad()
        self.temperature_loss.backward()
        self.temperature_optimizer.step()

    def sync_target_network(self):
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

    def __str__(self) -> str:
        from rlrl.utils.dict2colorized_string import dict2colorized_string

        title = "============== Soft Actor Critic =============="
        return dict2colorized_string(
            title,
            {
                "Policy": {"Network": self.q1, "Optimaizer": self.q1_optimizer},
                "Q Functions": {
                    "Q1": {"Network": self.q1, "Optimaizer": self.q1_optimizer},
                    "Q2": {"Network": self.q2, "Optimaizer": self.q2_optimizer},
                },
                "Temperature": {
                    "Network": self.temperature_holder,
                    "Optimizer": self.temperature_optimizer,
                },
                "gamma": self.gamma,
                "target entropy": self.target_entropy,
                "batch size": self.batch_size,
                "replay buffer capacity": self.replay_buffer.capacity,
                "device": self.device,
            },
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
