import copy
from typing import Optional, Type

from rlrl.agents.agent_base import AgentBase
from rlrl.replay_buffers.replay_buffer import ReplayBuffer
from torch import nn
from torch.optim import Adam, Optimizer


class Td3Agent(AgentBase):
    def __init__(
        self,
        dim_state: int,
        dim_action: int,
        policy: Optional[nn.Module] = None,
        policy_optimizer_type: Type[Optimizer] = Adam,
        gamma: float = 0.99,
        tau: float = 5e-3,
        replay_buffer=ReplayBuffer(10 ** 6),
        batch_size: int = 256,
        num_random_act: int = 10000,
    ) -> None:
        super().__init__()

        if policy is None:
            NotImplementedError()
        else:
            self.policy = policy

        self.target_policy = copy.deepcopy(self.policy).eval().requires_grad_(False)
        self.target_q_func1 = copy.deepcopy(self.q_func1).eval().requires_grad_(False)
        self.target_q_func2 = copy.deepcopy(self.q_func2).eval().requires_grad_(False)

    def observe(self, *args, **kwargs) -> None:
        return super().observe(*args, **kwargs)

    def update_if_dataset_is_ready(self, *args, **kwargs):
        return super().update_if_dataset_is_ready(*args, **kwargs)
