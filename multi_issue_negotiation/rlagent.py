import math

import torch

from utils import get_utility, get_utility_with_discount

import numpy as np
from agent import Agent

class RLAgent(Agent):
    def __init__(self, max_round, name, device, policy,qf1,qf2,target_qf1,target_qf2,issue_num=3, algo='sac',):
        super().__init__(max_round=max_round, name=name)
        self.max_round = max_round
        self.issue_num = issue_num
        self.state_dim = 7
        self.action_dim = 1
        self.algo = algo
        self.actor=policy
        self.q1=qf1
        self.q2=qf2
        self.target_q1=target_qf1
        self.target_q2=target_qf2
        self.obs=None
        
    def load(self, filename, directory):
        self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, filename)))
        self.q1.load_state_dict(torch.load('%s/%s_critic1.pth' % (directory, filename)))
        self.q2.load_state_dict(torch.load('%s/%s_critic2.pth' % (directory, filename)))
        self.target_q1.load_state_dict(torch.load('%s/%s_critic1.pth' % (directory, filename)))
        self.target_q2.load_state_dict(torch.load('%s/%s_critic2.pth' % (directory, filename)))
        

    def train(self, offer_replay_buffer, iterations, batch_size, discount, tau, policy_noise, noise_clip, policy_freq):
        return
        
    def reset(self):
        super().reset()
        self.update_obs()
    def get_obs(self):
        return self.obs
    def save(self, directory):
        return
        
    def update_network(self,networks):
       
        self.actor=networks[0]
        self.qf1 = networks[1]
        self.qf2 = networks[2]
        self.target_qf1 = networks[3]
        self.target_qf2 = networks[4]
        #self.actor = networks[0]

    def update_obs(self):
        if len(self.utility_received) >= 3:
            utility_received_3 = self.utility_received[-3]
            utility_received_2 = self.utility_received[-2]
            utility_received_1 = self.utility_received[-1]
        elif len(self.utility_received) >= 2:
            utility_received_3 = 0
            utility_received_2 = self.utility_received[-2]
            utility_received_1 = self.utility_received[-1]
        elif len(self.utility_received) >= 1:

            utility_received_3 = 0
            utility_received_2 = 0
            utility_received_1 = self.utility_received[-1]
        else:
            utility_received_3 = 0
            utility_received_2 = 0
            utility_received_1 = 0
        
        if len(self.utility_proposed) >= 3:
            utility_proposed_3 = self.utility_proposed[-3]
            utility_proposed_2 = self.utility_proposed[-2]
            utility_proposed_1 = self.utility_proposed[-1]
        elif len(self.utility_proposed) >= 2:
            utility_proposed_3 = 1
            utility_proposed_2 = self.utility_proposed[-2]
            utility_proposed_1 = self.utility_proposed[-1]
        elif len(self.utility_proposed) >= 1:
            utility_proposed_3 = 1
            utility_proposed_2 = 1
            utility_proposed_1 = self.utility_proposed[-1]
        else:
            utility_proposed_3 = 1
            utility_proposed_2 = 1
            utility_proposed_1 = 1

        self.obs = [self.t / self.max_round] + [utility_received_3] + [utility_proposed_3] + [utility_received_2] + [utility_proposed_2] + [utility_received_1] + [utility_proposed_1]


    def receive(self, last_action):
        if last_action is not None:
            oppo_offer = last_action
            utility = get_utility(oppo_offer, self.prefer, self.condition, self.domain_type, self.issue_value)
            
            self.offer_received.append(oppo_offer)
            self.utility_received.append(utility)
            if self.domain_type == "DISCRETE":
                value_recv = []
                for i in range(len(oppo_offer)):
                    value_recv.append(self.issue_value[i][oppo_offer[i]])
                self.value_received.append(value_recv)
                self.update_oppo_issue_value_estimation(oppo_offer, self.relative_t)
            self.oppo_prefer_estimater()
            self.update_obs()
            rl_utility = 0.5 * (self.actor.get_action(np.array(self.obs))[0] + 1) * (1 - self.u_min) + self.u_min
            
            self.accept = (rl_utility <= utility and self.t <= self.max_round)#
            #self.accept=False
            #self.accept = (rl_utility <= utility and self.t <= self.max_round and self.t>0.6*self.max_round)#


    def act(self):
        action ,_= self.actor.get_action(np.array(self.obs))
        #action = self.offer_policy.select_action(np.array(self.obs))
        return action


    def get_mu_std(self, state):
        return self.offer_policy.actor.mu_std(state)

    def Q_values(self, state, action):
        # print("state.shape:", state.shape, "action.shape:", action.shape)
        return self.offer_policy.critic(state, action)

    def V_value(self, state):
        return self.offer_policy.V_network(state)

    
    def loadSharedLayersParameter(self, student_model_path, opponent_label=None):
        if self.algo == 'sac2':
            self.offer_policy.loadSharedLayersParameter(student_model_path, opponent_label)
        else:
            raise NotImplementedError
        
    def frozenSharedLayersParameter(self):
        if self.algo == 'sac2':
            self.offer_policy.frozenSharedLayersParameter()
        else:
            raise NotImplementedError

