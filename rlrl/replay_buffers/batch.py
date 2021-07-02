import torch


class TorchTensorBatch(object):
    def __init__(self, state, next_state, action, reward, terminal, device="cpu", **kwargs) -> None:
        super().__init__()
        self.state: torch.Tensor = torch.tensor(state)
        self.next_state: torch.Tensor = torch.tensor(next_state)
        self.action: torch.Tensor = torch.tensor(action)
        self.reward: torch.Tensor = torch.tensor(reward)
        self.terminal: torch.Tensor = torch.tensor(terminal)
        self.to(device)

    def to(self, device):
        self.state = self.state.to(device)
        self.next_state = self.next_state.to(device)
        self.action = self.action.to(device)
        self.reward = self.reward.to(device)
        self.terminal = self.terminal.to(device)
        return self
