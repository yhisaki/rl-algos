# RLRL

## Overview

I'm writing reinforcement learning code using [pfnet/pfrl](https://github.com/pfnet/pfrl) as a reference.

## Install Guide

- Install from source

  ```
  pip install -e .
  ```
  or
  ```
  pip install -e ".[dev]"
  ```

- Uninstall
  ```
  pip uninstall rlrl
  ```

## Implemented Alogorithms

- [Soft Actor-Critic (SAC)](https://arxiv.org/abs/1812.05905)
- [Trust Region Policy Optimization (TRPO)](https://arxiv.org/abs/1502.05477)
- [Average Reward TRPO (ATRPO)](https://arxiv.org/abs/2106.07329)
- [Twin Delayed DDPG (TD3)](https://arxiv.org/abs/1802.09477)

## Results

### mujoco

<img src=asset/Humanoid-v3.gif width=30%><img src=asset/Swimmer-v2.gif width=30%><img src=asset/HalfCheetah-v3.gif width=30%>
<img src=asset/Hopper-v2.gif width=30%><img src=asset/Ant-v3.gif width=30%><img src=asset/Walker2d-v3.gif width=30%>

### box2d

<img src=asset/BipedalWalker-v3.gif width=30%>
