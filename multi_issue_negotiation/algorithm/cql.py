
import gtimer as gt
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import torch.optim as optim


import rlkit.torch.pytorch_util as ptu
from  rlkit.core.eval_util import create_stats_ordered_dict
from  rlkit.torch.torch_rl_algorithm import TorchTrainer
from torch import autograd

class CQLTrainer(TorchTrainer):
    def __init__(
        self, 
            
        policy,
        qf1,
        qf2,
        target_qf1,
        target_qf2,
        state_dim,
        action_dim,

        save_model_dir,
        init_alpha=1.0,
        alpha_lr=3e-4,
        

        discount=0.99,
        reward_scale=1.0,

        policy_lr=1e-3,
        qf_lr=1e-3,
            
            
            
        optimizer_class=optim.Adam,

        soft_target_tau=1e-2,
        plotter=None,
        render_eval_paths=False,
        
        target_update_period=1,
            
            

        use_automatic_entropy_tuning=True,
        target_entropy=None,
        policy_eval_start=0,
        num_qs=2,

            
        min_q_version=3,
        temp=1.0,
        min_q_weight=1.0,

            
        max_q_backup=False,
        deterministic_backup=True,
        num_random=10,
        with_lagrange=False,
        lagrange_thresh=0.0,

        save_every_step=1000,
        save_models=1
    ):
        super().__init__()
        #self.env = env
        self.policy = policy
        self.qf1 = qf1
        self.qf2 = qf2
        self.target_qf1 = target_qf1
        self.target_qf2 = target_qf2
        self.save_model_dir=save_model_dir
        self.save_models=save_models
        self.save_every_step=save_every_step
        # self.weight_net = weight_net
        # self.w_activation = lambda x: torch.relu(x)
        #self.temperature = temperature
        
        self.soft_target_tau = soft_target_tau

        self.use_automatic_entropy_tuning = use_automatic_entropy_tuning
        if self.use_automatic_entropy_tuning:
            if target_entropy:
                self.target_entropy = target_entropy
            else:
                self.target_entropy = -np.prod(np.array(action_dim).shape).item()
            #self.log_alpha = ptu.zeros(1, requires_grad=True)
            self.log_alpha = ptu.tensor(np.log(init_alpha), requires_grad=True)
            self.alpha_optimizer = optimizer_class(
                [self.log_alpha],
                lr=policy_lr,
            )
        ''' 
        '''
        self.with_lagrange = with_lagrange
        if self.with_lagrange:
            self.target_action_gap = lagrange_thresh
            self.log_alpha_prime = ptu.zeros(1, requires_grad=True)
            self.alpha_prime_optimizer = optimizer_class(
                [self.log_alpha_prime],
                lr=qf_lr,
            )

        #self.plotter = plotter
        #self.render_eval_paths = render_eval_paths

        self.qf_criterion = nn.MSELoss()
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
        # self.weight_optimizer = optimizer_class(
        #     self.weight_net.parameters(),
        #     lr=weight_net_lr,
        # )

        self.discount = discount
        self.reward_scale = reward_scale
        self.eval_statistics = OrderedDict()
        self._n_train_steps_total = 0
        self._need_to_update_eval_statistics = True
        self.policy_eval_start = policy_eval_start
        
        self._current_epoch = 0
        self._policy_update_ctr = 0
        self._num_q_update_steps = 0
        self._num_policy_update_steps = 0
        self._num_policy_steps = 1
        
        self.num_qs = num_qs

        ## min Q
        self.temp = temp
        self.min_q_version = min_q_version
        self.min_q_weight = min_q_weight
        
        self.softmax = torch.nn.Softmax(dim=1)
        self.softplus = torch.nn.Softplus(beta=self.temp, threshold=20)

        self.max_q_backup = max_q_backup
        self.deterministic_backup = deterministic_backup
        self.num_random = num_random

        # For implementation on the 
        self.discrete = False
    
    def _get_tensor_values(self, obs, actions, network=None):
        action_shape = actions.shape[0]
        #actions.to("cuda")
        obs_shape = obs.shape[0]
        num_repeat = int (action_shape / obs_shape)
        obs_temp = obs.unsqueeze(1).repeat(1, num_repeat, 1).view(obs.shape[0] * num_repeat, obs.shape[1])
        preds = network(obs_temp, actions)
        preds = preds.view(obs.shape[0], num_repeat, 1)
        return preds

    def _get_policy_actions(self, obs, num_actions, network=None):
        obs_temp = obs.unsqueeze(1).repeat(1, num_actions, 1).view(obs.shape[0] * num_actions, obs.shape[1])
        dist=network(obs_temp)
        new_obs_actions,new_obs_log_pi = dist.rsample_and_logprob()#, reparameterize=True, return_log_prob=True,
        if not self.discrete:
            return new_obs_actions, new_obs_log_pi.view(obs.shape[0], num_actions, 1)
        else:
            return new_obs_actions

    def train_from_torch(self, batch):
        self._current_epoch += 1
        rewards = batch['rewards']
        terminals = batch['terminals']
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']

        """
        Policy and Alpha Loss
        """
        # new_obs_actions, policy_mean, policy_log_std, log_pi, *_ = self.policy(
        #     obs, reparameterize=True, return_log_prob=True,
        # )
        dist = self.policy(obs)
        new_obs_actions, log_pi= dist.rsample_and_logprob()
        
        if self.use_automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            alpha = self.log_alpha.exp()
        else:
            alpha_loss = 0
            alpha = 1

        
        
        q_new_actions = torch.min(
                            self.qf1(obs, new_obs_actions),
                            self.qf2(obs, new_obs_actions),
                        )

        policy_loss = (alpha*log_pi - q_new_actions).mean()

        
        
        """
        QF Loss
        """
        q1_pred = self.qf1(obs, actions)
        q2_pred = self.qf2(obs, actions)
        
        next_dist = self.policy(next_obs)
        new_next_actions, new_log_pi = next_dist.rsample_and_logprob()
        #next_obs, reparameterize=True, return_log_prob=True,
        
        # new_curr_actions, _, _, new_curr_log_pi, *_ = self.policy(
        #     obs, reparameterize=True, return_log_prob=True,
        # )

        if not self.max_q_backup:
            
            target_q_values = torch.min(
                self.target_qf1(next_obs, new_next_actions),
                self.target_qf2(next_obs, new_next_actions),
            )
            
            if not self.deterministic_backup:
                target_q_values = target_q_values - alpha * new_log_pi
        
        if self.max_q_backup:
            """when using max q backup"""
            next_actions_temp, _ = self._get_policy_actions(next_obs, num_actions=10, network=self.policy)
            target_qf1_values = self._get_tensor_values(next_obs, next_actions_temp, network=self.target_qf1).max(1)[0].view(-1, 1)
            target_qf2_values = self._get_tensor_values(next_obs, next_actions_temp, network=self.target_qf2).max(1)[0].view(-1, 1)
            target_q_values = torch.min(target_qf1_values, target_qf2_values)

        q_target = self.reward_scale * rewards + (1. - terminals) * self.discount * target_q_values
        q_target = q_target.detach()
            
        qf1_loss = self.qf_criterion(q1_pred, q_target)

        qf2_loss = self.qf_criterion(q2_pred, q_target)

        ## add CQL
        random_actions_tensor = torch.FloatTensor(q2_pred.shape[0] * self.num_random, actions.shape[-1]).uniform_(-1, 1).cuda()
        curr_actions_tensor, curr_log_pis = self._get_policy_actions(obs, num_actions=self.num_random, network=self.policy)
        new_curr_actions_tensor, new_log_pis = self._get_policy_actions(next_obs, num_actions=self.num_random, network=self.policy)
        q1_rand = self._get_tensor_values(obs, random_actions_tensor, network=self.qf1)
        q2_rand = self._get_tensor_values(obs, random_actions_tensor, network=self.qf2)
        q1_curr_actions = self._get_tensor_values(obs, curr_actions_tensor, network=self.qf1)
        q2_curr_actions = self._get_tensor_values(obs, curr_actions_tensor, network=self.qf2)
        q1_next_actions = self._get_tensor_values(obs, new_curr_actions_tensor, network=self.qf1)
        q2_next_actions = self._get_tensor_values(obs, new_curr_actions_tensor, network=self.qf2)

        cat_q1 = torch.cat(
            [q1_rand, q1_pred.unsqueeze(1), q1_next_actions, q1_curr_actions], 1
        )
        cat_q2 = torch.cat(
            [q2_rand, q2_pred.unsqueeze(1), q2_next_actions, q2_curr_actions], 1
        )
        std_q1 = torch.std(cat_q1, dim=1)
        std_q2 = torch.std(cat_q2, dim=1)

        if self.min_q_version == 3:
            # importance sammpled version
            random_density = np.log(0.5 ** curr_actions_tensor.shape[-1])
            cat_q1 = torch.cat(
                [q1_rand - random_density, q1_next_actions - new_log_pis.detach(), q1_curr_actions - curr_log_pis.detach()], 1
            )
            cat_q2 = torch.cat(
                [q2_rand - random_density, q2_next_actions - new_log_pis.detach(), q2_curr_actions - curr_log_pis.detach()], 1
            )
            
        min_qf1_loss = torch.logsumexp(cat_q1 / self.temp, dim=1,).mean() * self.min_q_weight * self.temp
        min_qf2_loss = torch.logsumexp(cat_q2 / self.temp, dim=1,).mean() * self.min_q_weight * self.temp
                    
        """Subtract the log likelihood of data"""
        min_qf1_loss = min_qf1_loss - q1_pred.mean() * self.min_q_weight
        min_qf2_loss = min_qf2_loss - q2_pred.mean() * self.min_q_weight
        
        if self.with_lagrange:
            alpha_prime = torch.clamp(self.log_alpha_prime.exp(), min=0.0, max=1000000.0)
            min_qf1_loss = alpha_prime * (min_qf1_loss - self.target_action_gap)
            min_qf2_loss = alpha_prime * (min_qf2_loss - self.target_action_gap)

            self.alpha_prime_optimizer.zero_grad()
            alpha_prime_loss = (-min_qf1_loss - min_qf2_loss)*0.5 
            alpha_prime_loss.backward(retain_graph=True)
            self.alpha_prime_optimizer.step()

        qf1_loss = qf1_loss + min_qf1_loss
        qf2_loss = qf2_loss + min_qf2_loss

        """
        Update networks
        """
        # Update the Q-functions iff 
        self._num_q_update_steps += 1
        self.qf1_optimizer.zero_grad()
        qf1_loss.backward(retain_graph=True)
        self.qf1_optimizer.step()

        
        self.qf2_optimizer.zero_grad()
        qf2_loss.backward(retain_graph=True)
        self.qf2_optimizer.step()

        self._num_policy_update_steps += 1
        self.policy_optimizer.zero_grad()
        policy_loss.backward(retain_graph=False)
        self.policy_optimizer.step()

        """
        Soft Updates
        """
        ptu.soft_update_from_to(
            self.qf1, self.target_qf1, self.soft_target_tau
        )
        if self.num_qs > 1:
            ptu.soft_update_from_to(
                self.qf2, self.target_qf2, self.soft_target_tau
            )
        """
        Save some statistics for eval
        """
        eval_statistics = OrderedDict()
        eval_statistics["iter"] = self._n_train_steps_total
        eval_statistics["Policy_Loss"] = policy_loss
        eval_statistics["Alpha_Loss"] = alpha_loss
        eval_statistics["Lagrange Alpha Loss"] = alpha_prime_loss
        eval_statistics["CQL1 Loss"] = min_qf1_loss
        eval_statistics["CQL2 Loss"] = min_qf2_loss

        eval_statistics["qf1_loss"] = qf1_loss
        eval_statistics["qf2_loss"] = qf2_loss
        eval_statistics["Alpha"] = alpha
        eval_statistics["Lagrange Alpha"] = alpha_prime
        eval_statistics["logp_pi"] = log_pi
        
        self._n_train_steps_total += 1
        #print(self._n_train_steps_total)

        if self.save_models==1:
            #save model
            if self._n_train_steps_total % self.save_every_step== 0:
                #self.save_model()
                print("step:",self._n_train_steps_total)
                # print("Episode: {} | Reward: {} | Polciy Loss: {} | Steps: {}".format(i, rewards, policy_loss, steps, ))
                self.save_model("iter" + str(self._n_train_steps_total), self.save_model_dir)
        return eval_statistics

    def save_model(self, filename, directory):
        torch.save(self.policy.state_dict(), '%s/%s_actor.pt' % (directory, filename))
        torch.save(self.qf1.state_dict(), '%s/%s_critic1.pt' % (directory, filename))
        torch.save(self.qf2.state_dict(), '%s/%s_critic2.pt' % (directory, filename))
        # 修改
        # torch.save(self.V_network.state_dict(), '%s/%s_V_network.pth' % (directory, filename))
        # torch.save(self.actor_optimizer.state_dict(), '%s/%s_actor_optimizer.pth' % (directory, filename))
        # torch.save(self.alpha_optimizer.state_dict(), '%s/%s_alpha_optimizer.pth' % (directory, filename))
        # torch.save(self.vf_optimizer.state_dict(), '%s/%s_vf_optimizer.pth' % (directory, filename))
        # torch.save(self.log_alpha, '%s/%s_log_alpha.pth' % (directory, filename))


    def get_diagnostics(self):
        return self.eval_statistics

    def end_epoch(self, epoch):
        self._need_to_update_eval_statistics = True

    @property
    def networks(self):
        base_list = [
            self.policy,
            self.qf1,
            self.qf2,
            self.target_qf1,
            self.target_qf2,
        ]
        return base_list

    def get_snapshot(self):
        return dict(
            policy=self.policy,
            qf1=self.qf1,
            qf2=self.qf2,
            target_qf1=self.target_qf1,
            target_qf2=self.target_qf2,
        )
    