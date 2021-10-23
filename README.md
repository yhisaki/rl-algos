# RLRL

## Concept

I'm writing reinforcement learning code using [pfnet/pfrl](https://github.com/pfnet/pfrl) as a reference.

## Install Guide


- Install from source
  ```
  pip install -e .
  ```

- Uninstall
  ```
  python3 setup.py develop -u
  ```

## Implemented Alogorithms

- [Soft Actor-Critic (SAC)](https://arxiv.org/abs/1812.05905)
- [Trust Region Policy Optimization (TRPO)](https://arxiv.org/abs/1502.05477)
- [Average Reward TRPO (ATRPO)](https://arxiv.org/abs/2106.07329)

## 実行結果

### BipedalWalker-v3 + SoftActorCritic

報酬推移と獲得方策

<img src=asset/sac-BipedalWalker-v3/result.png width=30%> <img src=asset/sac-BipedalWalker-v3/epi350.gif width=30%>

