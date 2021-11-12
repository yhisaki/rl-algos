from typing import List
import numpy as np

import torch


def _to_torch_tensor(arr, device):
    if isinstance(arr, torch.Tensor):
        return arr
    elif isinstance(arr, np.ndarray):
        return torch.tensor(arr, device=device)
    elif isinstance(arr, list):
        return torch.tensor(np.array(arr), device=device)
    else:
        raise RuntimeError()


class TrainingBatch(object):
    def __init__(
        self, state, next_state, action, reward, terminal, reset, device=None, **kwargs
    ) -> None:
        super().__init__()
        self.state = _to_torch_tensor(state, device)
        self.next_state = _to_torch_tensor(next_state, device)
        self.action = _to_torch_tensor(action, device)
        self.reward = _to_torch_tensor(reward, device)
        self.terminal = _to_torch_tensor(terminal, device)
        self.reset = _to_torch_tensor(reset, device)

    def __getitem__(self, idx):
        return TrainingBatch(
            state=self.state[idx],
            next_state=self.next_state[idx],
            action=self.action[idx],
            reward=self.reward[idx],
            terminal=self.terminal[idx],
            reset=self.reset[idx],
        )

    def __setitem__(self, idx, batch: "TrainingBatch"):
        self.state = batch.state[idx]
        self.next_state = batch.next_state[idx]
        self.action = batch.action[idx]
        self.reward = batch.reward[idx]
        self.terminal = batch.terminal[idx]
        self.reset = batch.reset[idx]

    def to(self, device):
        return TrainingBatch(
            state=self.state.to(device),
            next_state=self.next_state.to(device),
            action=self.action.to(device),
            reward=self.reward.to(device),
            terminal=self.terminal.to(device),
            reset=self.reset.to(device),
        )

    def __len__(self):
        batch_len = len(self.state)
        assert (
            batch_len
            == len(self.next_state)
            == len(self.action)
            == len(self.reward)
            == len(self.terminal)
            == len(self.reset)
        )
        return batch_len


class EpisodicTrainingBatch(object):
    def __init__(self, episodes: List[dict], device=None) -> None:
        super().__init__()
        self.num_episodes = len(episodes)
        len_each_episodes = [len(episode["state"]) for episode in episodes]
        self.indices = [0] + [
            sum(len_each_episodes[0 : i + 1]) for i in range(self.num_episodes)  # NOQA
        ]
        self.flatten = TrainingBatch(
            state=sum([episode["state"] for episode in episodes], []),
            next_state=sum([episode["next_state"] for episode in episodes], []),
            action=sum([episode["action"] for episode in episodes], []),
            reward=sum([episode["reward"] for episode in episodes], []),
            terminal=sum([episode["terminal"] for episode in episodes], []),
            reset=sum([episode["reset"] for episode in episodes], []),
            device=device,
        )

    def __getitem__(self, idx: int) -> TrainingBatch:
        return self.flatten[self.indices[idx] : self.indices[idx + 1]]  # NOQA

    def __setitem__(self, idx: int, batch: TrainingBatch):
        self.flatten[self.indices[idx] : self.indices[idx + 1]] = batch  # NOQA

    def __len__(self):
        return self.num_episodes

    # def __next__(self):
    #     pass
