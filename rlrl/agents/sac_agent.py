import copy
import numpy as np
from collections import namedtuple
import torch
import torch.nn.functional as F
from rlrl.q_funcs.clipped_double_qf import ClippedDoubleQF
from rlrl.policies.gaussian_policy import GaussianPolicy
from rlrl.utils.global_device import get_global_torch_device

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'mask'))


def calc_q_loss(
        batch: Transition,
        policy: GaussianPolicy,
        alpha,
        gamma: float,
        cdqf: ClippedDoubleQF,
        cdqf_target: ClippedDoubleQF):
  with torch.no_grad():
    ap, ap_log_prob = policy.sample(batch.next_state)
    q_sp_ap = cdqf_target.forward(batch.next_state, ap)
    q_target = batch.reward + batch.mask * gamma * (q_sp_ap.squeeze() - alpha * ap_log_prob)

  q0 = cdqf[0].forward(batch.state, batch.action).squeeze()
  q1 = cdqf[1].forward(batch.state, batch.action).squeeze()

  loss = F.mse_loss(q0, q_target) + F.mse_loss(q1, q_target)
  return loss


def calc_policy_loss(
    batch: Transition,
    policy: GaussianPolicy,
    alpha,
    cdq: ClippedDoubleQF
):
  action, action_log_prob = policy.sample(batch.state)
  q = cdq.forward(batch.state, action)
  loss = (alpha * action_log_prob - q.squeeze()).mean()
  return loss


def calc_temperature_loss(
    batch: Transition,
    policy: GaussianPolicy,
    alpha,
    target_entropy
):
  with torch.no_grad():
    _, action_log_prob = policy.sample(batch.state)
  loss = - (alpha.exp() * (action_log_prob + target_entropy)).mean()
  return loss

