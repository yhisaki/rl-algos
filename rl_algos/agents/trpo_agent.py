import collections
import random
from logging import Logger, getLogger
from typing import Iterable, Optional, Type, Union

import torch
import torch.nn.functional as F
from torch import autograd, cuda, distributions, nn
from torch.optim import Adam, Optimizer

from rl_algos.agents.agent_base import AgentBase, AttributeSavingMixin
from rl_algos.agents.gae import generalized_advantage_estimation
from rl_algos.buffers.batch import EpisodicTrainingBatch, TrainingBatch
from rl_algos.buffers.episode_buffer import EpisodeBuffer
from rl_algos.modules import ZScoreFilter, ortho_init
from rl_algos.modules.distributions import (
    GaussianHeadWithStateIndependentCovariance,
    StochasticHeadBase,
)
from rl_algos.utils import clear_if_maxlen_is_none, conjugate_gradient, mean_or_nan


def _hessian_vector_product(
    flat_grads: torch.Tensor, params: Iterable[torch.Tensor], vec: torch.Tensor
):
    """
    Compute hessian vector product efficiently by backprop.
    Before executing this function, the parameter gradient must be initialized.
    (zero_grad())
    """
    vec = vec.detach()
    torch.sum(flat_grads * vec).backward(retain_graph=True)
    grads = nn.utils.convert_parameters.parameters_to_vector(parameters_grad(params))
    return grads


def parameters_grad(params: Iterable[torch.Tensor]) -> Iterable[torch.Tensor]:
    """return the iterable of parameter's grad"""
    for p in params:
        yield p.grad


def _yield_minibatches(dataset, minibatch_size, num_epochs):
    assert dataset
    buf = []
    n = 0
    while n < len(dataset) * num_epochs:
        while len(buf) < minibatch_size:
            buf = random.sample(dataset, k=len(dataset)) + buf
        assert len(buf) >= minibatch_size
        yield buf[-minibatch_size:]
        n += minibatch_size
        buf = buf[:-minibatch_size]


class TRPO(AttributeSavingMixin, AgentBase):
    saved_attributes = ("policy", "vf", "vf_optimizer", "state_normalizer")

    def __init__(
        self,
        dim_state: int,
        dim_action: int,
        policy: Optional[nn.Module] = None,
        vf: Optional[nn.Module] = None,
        vf_optimizer_class: Type[Optimizer] = Adam,
        vf_optimizer_kwargs: dict = {},
        vf_epoch=3,
        vf_batch_size=64,
        update_interval: int = 5000,
        recurrent: bool = False,
        state_normalizer: Optional[ZScoreFilter] = None,
        gamma: float = 0.99,
        lambd: float = 0.97,
        entropy_coef: float = 0.01,
        max_kl: float = 0.01,
        line_search_max_backtrack: int = 10,
        conjugate_gradient_max_iter: int = 10,
        conjugate_gradient_damping=1e-2,
        device: Union[str, torch.device] = torch.device("cuda:0" if cuda.is_available() else "cpu"),
        calc_stats=True,
        value_stats_window=None,
        entropy_stats_window=None,
        kl_stats_window=None,
        logger: Logger = getLogger(__name__),
    ) -> None:
        super().__init__()

        if isinstance(device, str):
            device = torch.device(device)
        self.device = device
        logger.info(f"DEVICE: {self.device}")

        # initialize policy
        if policy is None:
            self.policy = nn.Sequential(
                ortho_init(nn.Linear(dim_state, 64), gain=1.0),
                nn.Tanh(),
                ortho_init(nn.Linear(64, 64), gain=1.0),
                nn.Tanh(),
                ortho_init(nn.Linear(64, dim_action), gain=1e-2),
                GaussianHeadWithStateIndependentCovariance(dim_action),
            ).to(self.device)
        else:
            self.policy = policy.to(self.device)

        self.policy_stochastic_head: StochasticHeadBase = self.policy[-1]
        assert isinstance(self.policy_stochastic_head, StochasticHeadBase)

        # initialize value function
        if vf is None:
            self.vf = nn.Sequential(
                ortho_init(nn.Linear(dim_state, 64), gain=1.0),
                nn.Tanh(),
                ortho_init(nn.Linear(64, 64), gain=1.0),
                nn.Tanh(),
                ortho_init(nn.Linear(64, 1), gain=1e-2),
            ).to(self.device)
        else:
            self.vf = vf.to(self.device)

        self.vf_optimizer = vf_optimizer_class(self.vf.parameters(), **vf_optimizer_kwargs)
        self.vf_batch_size = vf_batch_size
        self.vf_epoch = vf_epoch

        self.recurrent = recurrent
        self.state_normalizer = state_normalizer
        if self.state_normalizer is not None:
            self.state_normalizer.to(self.device)

        self.buffer = EpisodeBuffer()
        self.gamma = gamma  # discount factor
        self.lambd = lambd  # eligibility trace parameter
        self.standardize_advantages = True
        self.line_search_max_backtrack = line_search_max_backtrack
        self.conjugate_gradient_max_iter = conjugate_gradient_max_iter
        self.conjugate_gradient_damping = conjugate_gradient_damping
        self.entropy_coef = entropy_coef
        self.max_kl = max_kl
        self.logger = logger
        self.num_update = 0  # number of update
        self.update_interval = update_interval

        self.calc_stats = calc_stats
        if self.calc_stats:
            self.value_record = collections.deque(maxlen=value_stats_window)
            self.entropy_record = collections.deque(maxlen=entropy_stats_window)
            self.kl_record = collections.deque(maxlen=kl_stats_window)

    def observe(
        self,
        states,
        next_states,
        actions,
        rewards,
        terminals,
        resets,
        **kwargs,
    ):
        for id, (state, next_state, action, reward, terminal, reset) in enumerate(
            zip(
                states,
                next_states,
                actions,
                rewards,
                terminals,
                resets,
            )
        ):
            self.buffer.append(
                id,
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                terminal=terminal,
                reset=reset,
                **kwargs,
            )
        if self.training:
            self.update_if_dataset_is_ready()

    def act(self, state):
        state = torch.tensor(state, device=self.device, requires_grad=False)
        if self.state_normalizer is not None:
            state = self.state_normalizer(state, update=False)
        with torch.no_grad():
            if self.training:
                action_distrib: distributions.Distribution = self.policy(state)
                action: torch.Tensor = action_distrib.sample()
                if self.calc_stats:
                    self.entropy_record.extend(action_distrib.entropy().cpu().numpy())
                    value: torch.Tensor = self.vf(state)
                    self.value_record.extend(value.cpu().numpy())
            else:
                with self.policy_stochastic_head.deterministic():
                    action: torch.Tensor = self.policy(state)

            action = action.cpu().numpy()
        return action

    def update_if_dataset_is_ready(self):
        assert self.training
        self.just_updated = False
        if len(self.buffer) >= self.update_interval:
            self.just_updated = True
            self.num_update += 1
            self.logger.info(f"Update TRPO num: {self.num_update}")

            episodes = self.buffer.get_episodes()
            batch = EpisodicTrainingBatch(episodes, device=self.device)

            if self.state_normalizer:
                batch.flatten.state = self.state_normalizer(batch.flatten.state, update=True)
                batch.flatten.next_state = self.state_normalizer(
                    batch.flatten.next_state, update=False
                )
            adv, v_target = generalized_advantage_estimation(
                batch, gamma=self.gamma, lambd=self.lambd, vf=self.vf, device=self.device
            )
            self._update_policy(batch.flatten, adv)
            self._update_vf(batch.flatten, v_target)
            self.buffer.clear()

    def _update_policy(self, batch: TrainingBatch, adv: torch.Tensor):
        if self.standardize_advantages:
            std_adv, mean_adv = torch.std_mean(adv, unbiased=False)
            adv = (adv - mean_adv) / (std_adv + 1e-8)

        action_distrib: distributions.Distribution = self.policy(batch.state)
        # Distribution to compute KL div against
        with torch.no_grad():
            # torch.distributions.Distribution cannot be deepcopied
            action_distrib_old: distributions.Distribution = self.policy(batch.state)
            log_prob_old = action_distrib_old.log_prob(batch.action)

        gain = self._compute_gain(
            log_prob=action_distrib.log_prob(batch.action),
            log_prob_old=log_prob_old,
            entropy=action_distrib.entropy(),
            adv=adv,
        )

        full_step = self._compute_kl_constrained_step(
            action_distrib=action_distrib,
            action_distrib_old=action_distrib_old,
            gain=gain,
        )

        self._line_search(
            full_step=full_step,
            batch=batch,
            adv=adv,
            action_distrib_old=action_distrib_old,
            log_prob_old=log_prob_old,
            gain=gain,
        )

    def _compute_gain(self, log_prob, log_prob_old, entropy, adv):
        prob_ratio = torch.exp(log_prob - log_prob_old)
        mean_entropy = torch.mean(entropy)
        surrogate_gain = torch.mean(prob_ratio * adv)
        return surrogate_gain + self.entropy_coef * mean_entropy

    def _compute_kl_constrained_step(
        self,
        action_distrib,
        action_distrib_old,
        gain: torch.Tensor,
    ):
        kl: torch.Tensor = distributions.kl_divergence(action_distrib_old, action_distrib).mean()
        self.policy.zero_grad()
        kl_grads = nn.utils.convert_parameters.parameters_to_vector(
            autograd.grad(kl, self.policy.parameters(), create_graph=True)
        )
        # kl.backward(create_graph=True)
        # kl_grads = nn.utils.convert_parameters.parameters_to_vector(
        #     parameters_grad(self.policy.parameters())
        # )

        def fisher_vector_product_func(vec):
            vec = torch.as_tensor(vec)
            self.policy.zero_grad()
            fvp = _hessian_vector_product(kl_grads, self.policy.parameters(), vec)
            return fvp + self.conjugate_gradient_damping * vec

        self.policy.zero_grad()
        gain.backward(retain_graph=True)
        gain_grads = nn.utils.convert_parameters.parameters_to_vector(  # δgain/δparams
            parameters_grad(self.policy.parameters())
        )
        step_direction = conjugate_gradient(
            fisher_vector_product_func, gain_grads, max_iter=self.conjugate_gradient_max_iter
        )

        dId = float(step_direction.dot(fisher_vector_product_func(step_direction)))
        scale = (2.0 * self.max_kl / (dId + 1e-8)) ** 0.5
        return scale * step_direction

    def _line_search(
        self,
        full_step,
        batch: TrainingBatch,
        adv,
        action_distrib_old,
        log_prob_old,
        gain,
    ):
        """Do line search for a safe step size."""
        step_size = 1.0
        flat_params = nn.utils.convert_parameters.parameters_to_vector(
            self.policy.parameters()
        ).detach()

        states = batch.state

        actions = batch.action

        for i in range(self.line_search_max_backtrack + 1):
            self.logger.info(f"Line search iteration: {i} step size: {step_size}")
            new_flat_params = flat_params + step_size * full_step
            nn.utils.convert_parameters.vector_to_parameters(
                new_flat_params, self.policy.parameters()
            )
            with torch.no_grad():
                new_action_distrib: distributions.Distribution = self.policy(states)
                new_gain = self._compute_gain(
                    log_prob=new_action_distrib.log_prob(actions),
                    log_prob_old=log_prob_old,
                    entropy=new_action_distrib.entropy(),
                    adv=adv,
                )
                new_kl = torch.mean(
                    distributions.kl_divergence(action_distrib_old, new_action_distrib)
                )

            improve = float(new_gain) - float(gain)
            self.logger.info(f"Surrogate objective improve: {improve}")
            self.logger.info(f"KL divergence: {float(new_kl)}")
            if not torch.isfinite(new_gain):
                self.logger.info("Surrogate objective is not finite. Bakctracking...")
            elif not torch.isfinite(new_kl):
                self.logger.info("KL divergence is not finite. Bakctracking...")
            elif improve < 0:
                self.logger.info("Surrogate objective didn't improve. Bakctracking...")
            elif float(new_kl) > self.max_kl:
                self.logger.info("KL divergence exceeds max_kl. Bakctracking...")
            else:
                self.kl_record.append(float(new_kl.cpu()))
                break
            step_size *= 0.5
        else:
            self.logger.info(
                "Line search coundn't find a good step size. The policy was not updated."
            )
            nn.utils.convert_parameters.vector_to_parameters(flat_params, self.policy.parameters())

    def _update_vf(self, batch: TrainingBatch, v_target: torch.Tensor):
        len_batch = len(batch)

        assert len_batch >= self.vf_batch_size
        for minibatch_indices in _yield_minibatches(
            range(len_batch), self.vf_batch_size, self.vf_epoch
        ):
            state = batch.state[minibatch_indices]
            v_targ = v_target[minibatch_indices]
            v_pred = self.vf(state)
            vf_loss = F.mse_loss(v_pred, v_targ[..., None])
            self.vf.zero_grad()
            vf_loss.backward()
            self.vf_optimizer.step()

    def get_statistics(self):
        if self.calc_stats:
            stats = {
                "average_entropy": mean_or_nan(self.entropy_record),
                "average_kl": mean_or_nan(self.kl_record),
                "average_value": mean_or_nan(self.value_record),
            }
            clear_if_maxlen_is_none(self.entropy_record, self.kl_record, self.value_record)
            return stats
        else:
            self.logger.warning("get_statistics() is called even though the calc_stats is False.")
