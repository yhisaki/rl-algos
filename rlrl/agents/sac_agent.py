from collections import namedtuple
from rlrl.utils.get_module_device import get_module_device
import torch
from torch import nn
import torch.nn.functional as F
from rlrl.q_funcs.clipped_double_qf import ClippedDoubleQF
from rlrl.policies.squashed_gaussian_policy import SquashedGaussianPolicy
from rlrl.nn.contexts import evaluating

Transition = namedtuple("Transition", ("state", "action", "next_state", "reward", "mask"))


class TemperatureHolder(nn.Module):
    """Module that holds a temperature as a learnable value.

    Args:
        initial_log_temperature (float): Initial value of log(temperature).
    """

    def __init__(self, initial_log_temperature=0):
        super().__init__()
        self.log_temperature = nn.Parameter(
            torch.tensor(initial_log_temperature, dtype=torch.float32)
        )

    def forward(self):
        """Return a temperature as a torch.Tensor."""
        return torch.exp(self.log_temperature)

    def to_float(self):
        with torch.no_grad():
            return float(self())


def calc_q_loss(
    batch: Transition,
    policy: torch.nn.Module,
    alpha: torch.Tensor,
    gamma: float,
    cdqf: ClippedDoubleQF,
    cdqf_target: ClippedDoubleQF,
) -> torch.Tensor:
    """Evaluation function of Q function in SAC

    Args:
        batch (Transition): Replay Buffer
        policy (torch.nn.Module): Policy. policy.forward(s) returns torch.distribution
        alpha (torch.Tensor): temperature.
        gamma (float): discount rate
        cdqf (ClippedDoubleQF): clipped double q function
        cdqf_target (ClippedDoubleQF): target clipped double q function

    Returns:
        torch.Tensor: [description]
    """
    with torch.no_grad():
        next_action_distrib = policy(batch.next_state)
        next_action = next_action_distrib.sample()
        next_log_prob = next_action_distrib.log_prob(next_action)
        next_q = cdqf_target(batch.next_state, next_action)
        entropy_term = alpha * next_log_prob[..., None]
        q_target = batch.reward + batch.mask * gamma * torch.flatten(next_q - entropy_term)

    q0 = torch.flatten(cdqf[0](batch.state, batch.action))
    q1 = torch.flatten(cdqf[1](batch.state, batch.action))

    loss = F.mse_loss(q0, q_target) + F.mse_loss(q1, q_target)
    return loss


def calc_policy_loss(
    batch: Transition, policy: SquashedGaussianPolicy, alpha: torch.Tensor, cdq: ClippedDoubleQF
) -> torch.Tensor:

    action_distrib = policy(batch.state)
    action = action_distrib.rsample()
    log_prob = action_distrib.log_prob(action)
    q = cdq.forward(batch.state, action)
    loss = (alpha * log_prob - q.flatten()).mean()
    return loss


def calc_temperature_loss(
    batch: Transition, policy: SquashedGaussianPolicy, alpha: torch.Tensor, target_entropy: float
) -> torch.Tensor:

    with torch.no_grad():
        action_distrib = policy(batch.state)
        action = action_distrib.sample()
        log_prob = action_distrib.log_prob(action)

    loss = -(alpha * (log_prob + target_entropy).detach()).mean()
    return loss


def get_action_and_entropy(state, policy, dev=None):
    """sampling action from policy and state(numpy or list)

    Args:
        state (numpy or list): [description]
        policy ([type]): [description]
        dev ([type]): [description]

    Returns:
        [type]: [description]
    """
    if dev is None:
        dev = get_module_device(policy)
    with torch.no_grad(), evaluating(policy):
        policy_distrib = policy(torch.tensor(state, dtype=torch.float32, device=dev))
        action = policy_distrib.sample()
        action_log_prob = policy_distrib.log_prob(action)
        action_log_prob = action_log_prob.cpu().numpy()
        action = action.cpu().numpy()

    return action, -action_log_prob
