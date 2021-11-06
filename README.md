# RLRL

## Overview

I'm writing reinforcement learning code using [pfnet/pfrl](https://github.com/pfnet/pfrl) as a reference.

## Install Guide


- Install from source
  ```
  pip install -e .
  ```

- Install from git
  ```
  pip install git+https://github.com/yhisaki/rlrl.git
  ```
- Uninstall
  ```
  python3 setup.py develop -u
  ```

## Implemented Alogorithms

- [Soft Actor-Critic (SAC)](https://arxiv.org/abs/1812.05905)
- [Trust Region Policy Optimization (TRPO)](https://arxiv.org/abs/1502.05477)
- [Average Reward TRPO (ATRPO)](https://arxiv.org/abs/2106.07329)
- [Twin Delayed DDPG](https://arxiv.org/abs/1802.09477)

## Results

### mujoco
<img src=asset/Humanoid-v3.gif width=30%><img src=asset/Swimmer-v2.gif width=30%><img src=asset/HalfCheetah-v3.gif width=30%>
<img src=asset/Hopper-v2.gif width=30%><img src=asset/Ant-v3.gif width=30%><img src=asset/Walker2d-v3.gif width=30%>

### box2d
<img src=asset/BipedalWalker-v3.gif width=30%>

