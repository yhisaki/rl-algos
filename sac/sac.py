import torch
import torch.nn.functional as F
import gym
from ReplayMemory import ReplayMemory, Transition
from GaussianPolicy import GaussianPolicy
from QNetwork import QNetwork
from typing import Tuple


def CalcQLoss(batch: Transition,
              policy: GaussianPolicy, alpha, gamma: float,
              QN: Tuple[QNetwork, QNetwork], QNtarget: Tuple[QNetwork, QNetwork]):
  with torch.no_grad():
    next_action, next_action_log_prob = policy.sample(batch.next_state)
    Q_next_state_target = torch.min(
        QNtarget[0].forward(batch.next_state, next_action),
        QNtarget[1].forward(batch.next_state, next_action))
    V_next_state_target = Q_next_state_target.T[0] - alpha * next_action_log_prob
    Q_state_target = batch.reward + batch.mask * gamma * V_next_state_target

  Q0 = QN[0].forward(batch.state, batch.action).T[0]
  Q1 = QN[1].forward(batch.state, batch.action).T[0]

  QLoss = F.mse_loss(Q0, Q_state_target) + F.mse_loss(Q1, Q_state_target)
  return QLoss


def CalcPolicyLoss(batch: Transition,
                   policy: GaussianPolicy, alpha,
                   QN: Tuple[QNetwork, QNetwork]):

  action, action_log_prob = policy.sample(batch.state)
  Q = torch.min(QN[0].forward(batch.state, action),
                QN[1].forward(batch.state, action))
  return (alpha * action_log_prob - Q.T[0]).mean()


def CalcTemperatureLoss(batch: Transition,
                        policy: GaussianPolicy, alpha, target_entropy):
  _, action_log_prob = policy.sample(batch.state)
  alpha_loss = - (alpha.exp() * (action_log_prob.detach() + target_entropy)).mean()
  return alpha_loss