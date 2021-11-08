import collections
import copy
import logging
from typing import Optional, Type, Union

import torch
import torch.nn.functional as F
from torch import cuda, distributions, nn
from torch.optim import Adam, Optimizer

from rlrl.agents.agent_base import AgentBase, AttributeSavingMixin
from rlrl.buffers import ReplayBuffer, TrainingBatch
from rlrl.explorers import ExplorerBase, GaussianExplorer
from rlrl.modules import ConcatStateAction, evaluating
from rlrl.modules.distributions import DeterministicHead, StochasticHeadBase
from rlrl.utils import clear_if_maxlen_is_none, mean_or_nan, synchronize_parameters


def default_target_policy_smoothing_func(batch_action):
    """Add noises to actions for target policy smoothing."""
    noise = torch.clamp(0.2 * torch.randn_like(batch_action), -0.5, 0.5)
    return torch.clamp(batch_action + noise, -1, 1)


class Td3Agent(AttributeSavingMixin, AgentBase):
    def __init__(
        self,
        dim_state: int,
        dim_action: int,
        q1: Optional[nn.Module] = None,
        q2: Optional[nn.Module] = None,
        q_optimizer_class: Type[Optimizer] = Adam,
        q_optimizer_kwargs: dict = {},
        q_tau: float = 5e-3,
        policy: Optional[nn.Module] = None,
        policy_optimizer_class: Type[Optimizer] = Adam,
        policy_optimizer_kwargs: dict = {},
        policy_tau: float = 5e-3,
        policy_update_delay: int = 2,
        explorer: ExplorerBase = GaussianExplorer(0.1, -1, 1),
        gamma: float = 0.99,
        replay_buffer: ReplayBuffer = ReplayBuffer(10 ** 6),
        batch_size: int = 256,
        num_random_act: int = 10 ** 4,
        calc_stats: bool = True,
        q_stats_window=None,
        q_loss_stats_window=None,
        policy_loss_stats_window=None,
        logger: logging.Logger = logging.getLogger(__name__),
        device: Union[str, torch.device] = torch.device("cuda:0" if cuda.is_available() else "cpu"),
    ) -> None:
        super().__init__()
        if isinstance(device, str):
            device = torch.device(device)
        self.device = device

        self.dim_state = dim_state
        self.dim_action = dim_action
        self.gamma = gamma

        if policy is None:
            self.policy = nn.Sequential(
                nn.Linear(self.dim_state, 400),
                nn.ReLU(),
                nn.Linear(400, 300),
                nn.ReLU(),
                nn.Linear(300, self.dim_action),
                nn.Tanh(),
                DeterministicHead(),
            ).to(self.device)
        else:
            self.policy = policy

        self.policy_optimizer = policy_optimizer_class(
            self.policy.parameters(), **policy_optimizer_kwargs
        )
        self.policy_head: StochasticHeadBase = self.policy[-1]
        self.policy_target = copy.deepcopy(self.policy).eval().requires_grad_(False)
        self.policy_tau = policy_tau
        self.policy_update_delay = policy_update_delay

        # configure Q
        if q1 is None:
            self.q1 = nn.Sequential(
                ConcatStateAction(),
                nn.Linear(self.dim_state + self.dim_action, 400),
                nn.ReLU(),
                nn.Linear(400, 300),
                nn.ReLU(),
                nn.Linear(300, 1),
            ).to(self.device)
        else:
            self.q1 = q1.to(self.device)

        if q2 is None:
            self.q2 = nn.Sequential(
                ConcatStateAction(),
                nn.Linear(self.dim_state + self.dim_action, 400),
                nn.ReLU(),
                nn.Linear(400, 300),
                nn.ReLU(),
                nn.Linear(300, 1),
            ).to(self.device)
        else:
            self.q2 = q2.to(self.device)

        self.q1_target = copy.deepcopy(self.q1).eval().requires_grad_(False)
        self.q2_target = copy.deepcopy(self.q2).eval().requires_grad_(False)
        self.q_tau = q_tau

        self.q1_optimizer = q_optimizer_class(self.q1.parameters(), **q_optimizer_kwargs)
        self.q2_optimizer = q_optimizer_class(self.q2.parameters(), **q_optimizer_kwargs)

        self.replay_buffer = replay_buffer
        self.batch_size = batch_size
        self.num_random_act = num_random_act
        self.explorer = explorer

        self.t = 0
        self.num_policy_update = 0
        self.num_q_update = 0

        self.calc_stats = calc_stats

        if self.calc_stats:
            self.q1_record = collections.deque(maxlen=q_stats_window)
            self.q2_record = collections.deque(maxlen=q_stats_window)
            self.q1_loss_record = collections.deque(maxlen=q_loss_stats_window)
            self.q2_loss_record = collections.deque(maxlen=q_loss_stats_window)
            self.policy_loss_record = collections.deque(maxlen=policy_loss_stats_window)

        self.logger = logger

    def _sync_target_network(self):
        """Synchronize target network with current network."""
        synchronize_parameters(
            src=self.policy,
            dst=self.policy_target,
            method="soft",
            tau=self.policy_tau,
        )
        synchronize_parameters(
            src=self.q1,
            dst=self.q1_target,
            method="soft",
            tau=self.q_tau,
        )
        synchronize_parameters(
            src=self.q2,
            dst=self.q2_target,
            method="soft",
            tau=self.q_tau,
        )

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
                with self.policy_head.deterministic():
                    actions = self.policy(states)

        actions = actions.detach().cpu().numpy()
        return actions

    def update_if_dataset_is_ready(self):
        assert self.training
        self.just_updated = False
        if len(self.replay_buffer) > self.num_random_act:
            self.just_updated = True
            if self.num_q_update == 0:
                self.logger.info("Start Update")
            sampled = self.replay_buffer.sample(self.batch_size)
            batch = TrainingBatch(**sampled, device=self.device)
            self._update_q(batch)
            if self.num_q_update % self.policy_update_delay == 0:
                self._update_policy(batch)
                self._sync_target_network()

    def _update_q(self, batch: TrainingBatch):
        q1_loss, q2_loss = self.compute_q_loss(
            batch=batch,
        )
        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        self.q1_optimizer.step()

        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        self.q2_optimizer.step()

        self.num_q_update += 1

    def _update_policy(self, batch: TrainingBatch):
        actions = self.policy(batch.state).rsample()
        q = self.q1((batch.state, actions))
        policy_loss = -torch.mean(q)

        self.policy_loss_record.append(float(policy_loss))

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        self.num_policy_update += 1

    def compute_q_loss(self, batch: TrainingBatch):
        with torch.no_grad(), evaluating(self.policy_target, self.q1_target, self.q2_target):
            next_actions: torch.Tensor = default_target_policy_smoothing_func(
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

        q1_loss = F.mse_loss(q1_pred, q_target)
        q2_loss = F.mse_loss(q2_pred, q_target)

        if self.calc_stats:
            if self.q1_record is not None:
                self.q1_record.extend(q1_pred.detach().cpu().numpy())
            if self.q2_record is not None:
                self.q2_record.extend(q2_pred.detach().cpu().numpy())
            if self.q1_loss_record is not None:
                self.q1_loss_record.append(float(q1_loss))
            if self.q2_loss_record is not None:
                self.q2_loss_record.append(float(q2_loss))

        return q1_loss, q2_loss

    def get_statistics(self) -> dict:
        if self.calc_stats:
            stats = {
                "average_q1": mean_or_nan(self.q1_record),
                "average_q2": mean_or_nan(self.q2_record),
                "average_q1_loss": mean_or_nan(self.q1_loss_record),
                "average_q2_loss": mean_or_nan(self.q2_loss_record),
                "average_policy_loss": mean_or_nan(self.policy_loss_record),
            }

            clear_if_maxlen_is_none(
                self.q1_record,
                self.q2_record,
                self.q1_loss_record,
                self.q2_loss_record,
                self.policy_loss_record,
            )

            return stats
        else:
            self.logger.warning("get_statistics() is called even though the calc_stats is False.")
