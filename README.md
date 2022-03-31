# Reinforcement Learning Algorithms

## Overview

I'm writing reinforcement learning code using [pfnet/pfrl](https://github.com/pfnet/pfrl) as a reference.

## Install

This package is managed using [poetry](https://python-poetry.org/).
You can install the dependencies by executing the following command.
See the [pyproject.toml](./pyproject.toml) for the detail of dependencies.
```
poetry install
```

If you excute examples, you should install [MuJoCo](https://mujoco.org/).

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
