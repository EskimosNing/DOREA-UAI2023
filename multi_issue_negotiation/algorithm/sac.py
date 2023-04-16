from collections import OrderedDict, namedtuple
from typing import Tuple

import numpy as np
import torch
import torch.optim as optim
from rlkit.core.loss import LossFunction, LossStatistics
from torch import nn as nn

from rlkit.launchers import conf

import rlkit.torch.pytorch_util as ptu
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.torch.torch_rl_algorithm import TorchTrainer
from rlkit.core.logging import add_prefix
import gtimer as gt

SACLosses = namedtuple(
    "SACLosses",
    "policy_loss qf1_loss qf2_loss alpha_loss",
)


class SACTrainer(TorchTrainer, LossFunction):
    def __init__(
        self,
        policy,
        qf1,
        qf2,
        target_qf1,
        target_qf2,
        save_model_dir,
        save_every_step,
        policy_eval_start,

        state_dim=7,
        action_dim=1,
        save_models=1,
        discount=0.99,
        reward_scale=1.0,
        policy_lr=1e-3,
        qf_lr=1e-3,
        alpha_lr=3e-4,
        init_alpha=0.3,
        optimizer_class=optim.Adam,
        soft_target_tau=1e-2,
        target_update_period=1,
        plotter=None,
        render_eval_paths=False,
        use_automatic_entropy_tuning=True,
        target_entropy=None,
        separate_buffers=False,
        save_checkpoints=True,
    ):
        super().__init__()

        self.policy = policy
        self.qf1 = qf1
        self.qf2 = qf2
        self.target_qf1 = target_qf1
        self.target_qf2 = target_qf2
        self.soft_target_tau = soft_target_tau
        self.target_update_period = target_update_period
        self.save_every_step=save_every_step
        self.save_checkpoints = save_checkpoints
        self.save_model_dir=save_model_dir
        self.save_every_step=save_every_step
        self.state_dim = state_dim,
        self.action_dim = action_dim,
        self.save_models = save_models,
        self.policy_eval_start=policy_eval_start

        self.use_automatic_entropy_tuning = use_automatic_entropy_tuning
        if self.use_automatic_entropy_tuning:
            if target_entropy is None:
                # Use heuristic value from SAC paper
                self.target_entropy = -np.prod(np.array(action_dim).shape).item()
            else:
                self.target_entropy = target_entropy
            # self.log_alpha = ptu.zeros(1, requires_grad=True)
            self.log_alpha = ptu.tensor([np.log(init_alpha)], requires_grad=True)
            self.alpha_optimizer = optimizer_class(
                [self.log_alpha],
                lr=alpha_lr,
            )

        self.plotter = plotter
        self.render_eval_paths = render_eval_paths

        self.qf_criterion = nn.MSELoss(reduction="none")
        self.vf_criterion = nn.MSELoss()

        self.policy_optimizer = optimizer_class(
            self.policy.parameters(),
            lr=policy_lr,
        )
        self.qf1_optimizer = optimizer_class(
            self.qf1.parameters(),
            lr=qf_lr,
        )
        self.qf2_optimizer = optimizer_class(
            self.qf2.parameters(),
            lr=qf_lr,
        )

        self.discount = discount
        self.reward_scale = reward_scale
        self._n_train_steps_total = 0
        self._need_to_update_eval_statistics = True
        self.eval_statistics = OrderedDict()


        self.separate_buffers = separate_buffers

    def train_from_torch(self, batch):
        gt.blank_stamp()
        losses, stats = self.compute_loss(
            batch,
            skip_statistics=not self._need_to_update_eval_statistics,
        )
        """
        Update networks
        """
        if self.use_automatic_entropy_tuning:
            self.alpha_optimizer.zero_grad()
            losses.alpha_loss.backward()
            self.alpha_optimizer.step()

        self.policy_optimizer.zero_grad()
        losses.policy_loss.backward()
        self.policy_optimizer.step()

        self.qf1_optimizer.zero_grad()
        losses.qf1_loss.backward()
        self.qf1_optimizer.step()

        self.qf2_optimizer.zero_grad()
        losses.qf2_loss.backward()
        self.qf2_optimizer.step()

        self._n_train_steps_total += 1


        self.try_update_target_networks()
        if self._need_to_update_eval_statistics:
            self.eval_statistics = stats
            # Compute statistics using only one batch per epoch
            self._need_to_update_eval_statistics = False
        gt.stamp("sac training", unique=False)

        if self._n_train_steps_total % 1000 == 0 and self.save_checkpoints:
            # save model
            #if self._n_train_steps_total % self.save_every_step == 0:
                # self.save_model()
            print("step:", self._n_train_steps_total)
                # print("Episode: {} | Reward: {} | Polciy Loss: {} | Steps: {}".format(i, rewards, policy_loss, steps, ))
            self.save_model("iter" + str(self._n_train_steps_total), self.save_model_dir)
            #self.save(self._n_train_steps_total)

        return stats

    def try_update_target_networks(self):
        if self._n_train_steps_total % self.target_update_period == 0:
            self.update_target_networks()

    def update_target_networks(self):
        ptu.soft_update_from_to(self.qf1, self.target_qf1, self.soft_target_tau)
        ptu.soft_update_from_to(self.qf2, self.target_qf2, self.soft_target_tau)

    def compute_loss(
        self,
        batch,
        skip_statistics=False,
    ) -> Tuple[SACLosses, LossStatistics]:
        rewards = batch["rewards"]
        terminals = batch["terminals"]
        obs = batch["observations"]
        actions = batch["actions"]
        next_obs = batch["next_observations"]
        batch_size = obs.shape[0]

        """
        Policy and Alpha Loss
        """
        dist = self.policy(obs)
        new_obs_actions, log_pi = dist.rsample_and_logprob()
        log_pi = log_pi.unsqueeze(-1)
        if self.use_automatic_entropy_tuning:
            alpha_loss = -(
                self.log_alpha * (log_pi + self.target_entropy).detach()
            ).mean()
            alpha = self.log_alpha.exp()
        else:
            alpha_loss = 0
            alpha = 1

        q_new_actions = torch.min(
            self.qf1(obs, new_obs_actions),
            self.qf2(obs, new_obs_actions),
        )
        policy_loss = (alpha * log_pi - q_new_actions).mean()

        """
        QF Loss
        """
        q1_pred = self.qf1(obs, actions)
        q2_pred = self.qf2(obs, actions)
        next_dist = self.policy(next_obs)
        new_next_actions, new_log_pi = next_dist.rsample_and_logprob()
        new_log_pi = new_log_pi.unsqueeze(-1)
        target_q_values = torch.min(
            self.target_qf1(next_obs, new_next_actions),
            self.target_qf2(next_obs, new_next_actions),
        )

        target_q_values = target_q_values - alpha * new_log_pi

        q_target = (
            self.reward_scale * rewards
            + (1.0 - terminals) * self.discount * target_q_values
        )
        qf1_loss_ = self.qf_criterion(q1_pred, q_target.detach())
        qf2_loss_ = self.qf_criterion(q2_pred, q_target.detach())
        qf1_loss = qf1_loss_.mean()
        qf2_loss = qf2_loss_.mean()

        """
        Save some statistics for eval
        """
        eval_statistics = OrderedDict()
        eval_statistics["iter"]=self._n_train_steps_total
        eval_statistics["Policy_Loss"] = policy_loss
        eval_statistics["Alpha_Loss"] = alpha_loss
        eval_statistics["qf1_loss"] = qf1_loss
        eval_statistics["qf2_loss"] = qf2_loss
        eval_statistics["Alpha"] = alpha
        eval_statistics["logp_pi"] = log_pi
        loss = SACLosses(
            policy_loss=policy_loss,
            qf1_loss=qf1_loss,
            qf2_loss=qf2_loss,
            alpha_loss=alpha_loss,
        )

        return loss, eval_statistics

    def get_diagnostics(self):
        stats = super().get_diagnostics()
        stats.update(self.eval_statistics)
        return stats

    def end_epoch(self, epoch):
        self._need_to_update_eval_statistics = True

    def save_model(self, filename, directory):
        torch.save(self.policy.state_dict(), '%s/%s_actor.pt' % (directory, filename))
        torch.save(self.qf1.state_dict(), '%s/%s_critic1.pt' % (directory, filename))
        torch.save(self.qf2.state_dict(), '%s/%s_critic2.pt' % (directory, filename))

    @property
    def networks(self):
        return [
            self.policy,
            self.qf1,
            self.qf2,
            self.target_qf1,
            self.target_qf2,
        ]

    @property
    def optimizers(self):
        return [
            self.alpha_optimizer,
            self.qf1_optimizer,
            self.qf2_optimizer,
            self.policy_optimizer,
        ]

    def get_snapshot(self):
        return dict(
            policy=self.policy,
            qf1=self.qf1,
            qf2=self.qf2,
            target_qf1=self.target_qf1,
            target_qf2=self.target_qf2,
        )
