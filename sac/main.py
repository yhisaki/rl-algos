import gym
from gym import wrappers
from gym.wrappers.monitoring import video_recorder
import torch
from torch.optim import Adam
from ReplayMemory import ReplayMemory, Transition
from GaussianPolicy import GaussianPolicy
from QNetwork import QNetwork, SynchronizeQNetwork
from SacLoss import CalcQLoss, CalcPolicyLoss, CalcTemperatureLoss
from VideoUtil import GymVideo
from itertools import chain
import matplotlib.pyplot as plt

# gym.logger.set_level(40)

# ENV_ID = 'MountainCarContinuous-v0'
ENV_ID = 'BipedalWalker-v3'

# Parameters of SAC
NUM_EPISODES = 5000
REPLAY_MEMORY_CAPACITY: int = 1E+6
T_INIT = 10000
HIDDEN_DIM = 256
BATCH_SIZE = 256
LR = 1E-3
GAMMA = 0.99
TAU = 0.005

# device
dev = torch.device("cuda")

# Make enviroment
env = gym.make(ENV_ID)
max_episode_steps = env._max_episode_steps
vid = GymVideo()
# vid = video_recorder.VideoRecorder(env, path="./video/" + ENV_ID + ".mp4")
# env = wrappers.Monitor(env, "./video/" + ENV_ID, video_callable=(lambda ep: ep % 100 == 0), force=True)

# Get Dimension
dim_state = env.observation_space.shape[0]
dim_action = env.action_space.shape[0]

# ReplayMemory
D = ReplayMemory(REPLAY_MEMORY_CAPACITY)

# Soft Q Function
q1 = QNetwork(dim_state, dim_action, HIDDEN_DIM).to(dev)
q1_target = QNetwork(dim_state, dim_action, HIDDEN_DIM).to(dev)

q2 = QNetwork(dim_state, dim_action, HIDDEN_DIM).to(dev)
q2_target = QNetwork(dim_state, dim_action, HIDDEN_DIM).to(dev)

SynchronizeQNetwork(q1, q1_target)
SynchronizeQNetwork(q2, q2_target)

# Policy
policy = GaussianPolicy(dev, dim_state, dim_action, HIDDEN_DIM,
                        env.action_space.high, env.action_space.low)

qnOptimizer = Adam(chain(q1.parameters(), q2.parameters()), lr=LR)
policyOptimizer = Adam(policy.parameters(), lr=LR)

# Alpha
log_alpha = torch.zeros(1, requires_grad=True)
alpha = log_alpha.exp().to(dev)
alphaOptimizer = Adam([log_alpha], lr=LR)
target_entropy = - dim_action

# Record
reward_record = []

try:

  for epi in range(NUM_EPISODES):
    state = env.reset()
    reward_sum = 0
    episode_step = 0
    action_log_prob_sum = 0
    while True:
      if len(D) < T_INIT:
        action = env.action_space.sample()
        state_next, reward, terminal, _ = env.step(action)
        D.store(state, action, state_next, reward, terminal)
      else:
        action, action_log_prob = policy.sample(state)
        action_log_prob_sum += action_log_prob.exp().to('cpu').detach().numpy()
        state_next, reward, done, _ = env.step(action.to('cpu').numpy())
        terminal = done if episode_step < max_episode_steps else False
        D.store(state, action.to('cpu').numpy(), state_next, reward, terminal)

        Dsub = D.sample_batch(BATCH_SIZE, "FloatTensor", dev)
        JQ = CalcQLoss(Dsub, policy, alpha, GAMMA,
                       (q1, q2), (q1_target, q2_target))

        qnOptimizer.zero_grad()
        JQ.backward()
        qnOptimizer.step()

        Jpi = CalcPolicyLoss(Dsub, policy, alpha, (q1, q2))
        policyOptimizer.zero_grad()
        Jpi.backward()
        policyOptimizer.step()

        Jalpha = CalcTemperatureLoss(Dsub, policy, log_alpha, target_entropy)
        alphaOptimizer.zero_grad()
        Jalpha.backward()
        alphaOptimizer.step()
        alpha = log_alpha.exp().to(dev)

        SynchronizeQNetwork(q1, q1_target, TAU)
        SynchronizeQNetwork(q2, q2_target, TAU)

      reward_sum += reward
      episode_step += 1
      state = state_next
      if epi % 50 == 0: vid.append_frame(env.render(mode="rgb_array"))
      if terminal:
        print(
            f"epi = {epi}, rsum = {reward_sum}, action_prob={action_log_prob_sum}, alpha = {alpha[0].to('cpu').detach().numpy()}")
        reward_record.append(reward_sum)
        if len(vid) != 0 : vid.save_video("video/" + ENV_ID + "_EPI" + str(epi) + ".gif" )
        break

except KeyboardInterrupt:
  # vid.close()
  env.close()
  pass

plt.close()
plt.plot(reward_record)
plt.show()
