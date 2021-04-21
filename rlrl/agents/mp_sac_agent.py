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

def calc_q_loss2(
        batch: Transition,
        policy: GaussianPolicy,
        another_policy: GaussianPolicy,
        beta1,
        beta2,
        gamma: float,
        cdqf: ClippedDoubleQF,
        cdqf_target: ClippedDoubleQF):
  with torch.no_grad():
    ap, ap_log_prob = policy.sample(batch.next_state)
    another_policy_entropy = another_policy.get_action_entropy(batch.next_state, ap)
    q_sp_ap = cdqf_target.forward(batch.next_state, ap)
    q_target = batch.reward + batch.mask * gamma * (q_sp_ap.squeeze() - beta1 * ap_log_prob - beta2 * another_policy_entropy)

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

def calc_policy_loss2(
    batch: Transition,
    policy: GaussianPolicy,
    another_policy: GaussianPolicy,
    beta1,
    beta2,
    cdq: ClippedDoubleQF
):
  action, action_log_prob = policy.sample(batch.state)
  another_policy_entropy = another_policy.get_action_entropy(batch.state, action)
  q = cdq.forward(batch.state, action)
  loss = (beta1 * action_log_prob + beta2 * another_policy_entropy - q.squeeze()).mean()
  return loss

def calc_temperature_loss(
    batch: Transition,
    policy: GaussianPolicy,
    log_alpha,
    target_entropy
):
  _, action_log_prob = policy.sample(batch.state)
  loss = - (log_alpha.exp().to(get_global_torch_device()) * (action_log_prob + target_entropy)).mean()
  return loss

