import gym
import torch
from torch.optim import Adam
from ReplayMemory import ReplayMemory, Transition
from GaussianPolicy import GaussianPolicy
from QNetwork import QNetwork, SynchronizeQNetwork
from sac import CalcQLoss, CalcPolicyLoss, CalcTemperatureLoss
from itertools import chain

gym.logger.set_level(40)

ENV_ID = 'MountainCarContinuous-v0'

# Parameters of SAC
NUM_EPISODES = 1000
REPLAY_MEMORY_CAPACITY: int = 1E+4
T_INIT = 300
HIDDEN_DIM = 256
BATCH_SIZE = 256
LR = 1E-3
GAMMA = 0.99
TAU = 0.005

# device
device = torch.device("cpu")

# Make enviroment
env = gym.make(ENV_ID)

# Get Dimension
dim_state = env.observation_space.shape[0]
dim_action = env.action_space.shape[0]

# ReplayMemory
D = ReplayMemory(REPLAY_MEMORY_CAPACITY)

# Soft Q Function
q1 = QNetwork(dim_state, dim_action, HIDDEN_DIM)
q1_target = QNetwork(dim_state, dim_action, HIDDEN_DIM)

q2 = QNetwork(dim_state, dim_action, HIDDEN_DIM)
q2_target = QNetwork(dim_state, dim_action, HIDDEN_DIM)

# Policy
policy = GaussianPolicy(dim_state, dim_action, HIDDEN_DIM,
                        env.action_space.high, env.action_space.low)

qnOptimizer = Adam(chain(q1.parameters(), q2.parameters()), lr=LR)
policyOptimizer = Adam(policy.parameters(), lr=LR)

# Policy
log_alpha = torch.zeros(1, requires_grad=True)
alpha = log_alpha.exp()
alphaOptimizer = Adam([log_alpha], lr=LR)
target_entropy = - dim_action

for epi in range(NUM_EPISODES):
  state = env.reset()
  reward_sum = 0
  while True:
    if len(D) < T_INIT:
      action = env.action_space.sample()
      state_next, reward, terminal, _ = env.step(action)
      reward_sum += reward
      D.store(state, action, state_next, reward, terminal)
    else:
      action, action_log_prob = policy.sample(state)
      reward_sum += reward
      state_next, reward, terminal, _ = env.step(action.numpy())
      D.store(state, action.numpy(), state_next, reward, terminal)

      Dsub = D.sample_batch(BATCH_SIZE, "FloatTensor", device)
      JQ = CalcQLoss(Dsub, policy, alpha, GAMMA, (q1, q2), (q1_target, q2_target))

      qnOptimizer.zero_grad()
      JQ.backward()
      qnOptimizer.step()

      Jpi = CalcPolicyLoss(Dsub, policy, alpha, (q1,q2))
      policyOptimizer.zero_grad()
      Jpi.backward()
      policyOptimizer.step()

      Jalpha = CalcTemperatureLoss(Dsub, policy, log_alpha, target_entropy)
      alphaOptimizer.zero_grad()
      Jalpha.backward()
      alphaOptimizer.step()
      alpha = log_alpha.exp()

      SynchronizeQNetwork(q1, q1_target, TAU)
      SynchronizeQNetwork(q2, q2_target, TAU)

    state = state_next

    if terminal:
      print(f"epi = {epi}, rsum = {reward_sum}, alpha = {alpha}")
      break

