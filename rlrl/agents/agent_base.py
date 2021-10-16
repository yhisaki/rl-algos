import contextlib
import os
from abc import ABCMeta, abstractmethod, abstractproperty
from typing import Any, List, Tuple

# import cloudpickle
# from gym.core import Env
import torch
from torch import cuda, nn


class AgentBase(object, metaclass=ABCMeta):
    """Abstract agent class."""

    training = True

    @abstractmethod
    def act(self, *args, **kwargs) -> Any:
        raise NotImplementedError()

    @abstractmethod
    def observe(self, *args, **kwargs) -> None:
        """
        Observe consequences of the last action.(e.g. state, next_state, action, reward, terminal)
        """
        raise NotImplementedError()

    @abstractmethod
    def update_if_dataset_is_ready(self, *args, **kwargs) -> Any:
        """
        Update the agent.(e.g. policy, q_function, ...)
        """
        raise NotImplementedError()

    @abstractmethod
    def save(self, dirname: str) -> None:
        """Save internal states.

        Returns:
            None
        """
        pass

    @abstractmethod
    def load(self, dirname: str) -> None:
        """Load internal states.

        Returns:
            None
        """
        pass

    @contextlib.contextmanager
    def eval_mode(self):
        orig_mode = self.training
        try:
            self.training = False
            yield
        finally:
            self.training = orig_mode


class BatchAgentBase(object, metaclass=ABCMeta):
    """Abstract agent class."""

    training = True

    @abstractmethod
    def batch_act(self, *args, **kwargs) -> Any:
        raise NotImplementedError()

    @abstractmethod
    def batch_observe(self, *args, **kwargs) -> None:
        """
        Observe consequences of the last action.(e.g. state, next_state, action, reward, terminal)
        """
        raise NotImplementedError()


class AttributeSavingMixin(object):
    """Mixin that provides save and load functionalities."""

    @abstractproperty
    def saved_attributes(self) -> Tuple[str, ...]:
        """Specify attribute names to save or load as a tuple of str."""
        pass

    def save(self, dirname: str) -> None:
        """Save internal states."""
        self.__save(dirname, [])

    def __save(self, dirname: str, ancestors: List[Any]):
        os.makedirs(dirname, exist_ok=True)
        ancestors.append(self)
        for attr in self.saved_attributes:
            assert hasattr(self, attr)
            attr_value = getattr(self, attr)
            if attr_value is None:
                continue
            if isinstance(attr_value, AttributeSavingMixin):
                assert not any(
                    attr_value is ancestor for ancestor in ancestors
                ), "Avoid an infinite loop"
                attr_value.__save(os.path.join(dirname, attr), ancestors)
            else:
                if isinstance(
                    attr_value,
                    (nn.parallel.DistributedDataParallel, nn.DataParallel),
                ):
                    attr_value = attr_value.module
                torch.save(attr_value.state_dict(), os.path.join(dirname, "{}.pt".format(attr)))
        ancestors.pop()

    def load(self, dirname: str) -> None:
        """Load internal states."""
        self.__load(dirname, [])

    def __load(self, dirname: str, ancestors: List[Any]) -> None:
        map_location = torch.device("cpu") if not cuda.is_available() else None
        ancestors.append(self)
        for attr in self.saved_attributes:
            assert hasattr(self, attr)
            attr_value = getattr(self, attr)
            if attr_value is None:
                continue
            if isinstance(attr_value, AttributeSavingMixin):
                assert not any(
                    attr_value is ancestor for ancestor in ancestors
                ), "Avoid an infinite loop"
                attr_value.load(os.path.join(dirname, attr))
            else:
                if isinstance(
                    attr_value,
                    (nn.parallel.DistributedDataParallel, nn.DataParallel),
                ):
                    attr_value = attr_value.module
                attr_value.load_state_dict(
                    torch.load(os.path.join(dirname, "{}.pt".format(attr)), map_location)
                )
        ancestors.pop()
