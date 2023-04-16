import os
import pickle

import argparse



import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np


import rlkit.torch.pytorch_util as ptu

from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
from rlkit.data_management.prioritized_replay_buffer import PriorityReplayBuffer

from rlkit.launchers.launcher_util import setup_logger
from rlkit.samplers.data_collector import MdpPathCollector

from rlkit.torch.sac.policies import TanhGaussianPolicy, MakeDeterministic
from  algorithm.fintune_trainer import OursTrainer

from rlkit.torch.networks import ConcatMlp

from rlkit.torch.torch_rl_algorithm import (
    TorchOurRLAlgorithm,
)
from rlkit.torch.distributions import TanhNormal
from rlkit.torch.core import torch_ify, elem_or_tuple_to_numpy


from utils import get_utility
from utils import ReplayBuffer
from rlagent import RLAgent
from negotiation import Negotiation
from agent36 import Agent36
from AgreeableAgent2018 import AgreeableAgent2018
from atlas3 import Atlas3
from ponpokoagent import PonPokoAgent
from parscat import ParsCat
from ParsAgent import ParsAgent
from theFawkes import TheFawkes
from agentlg import AgentLG
from omacagent import OMAC
from CUHKAgent import CUHKAgent
from HardHeaded import HardHeadedAgent
from YXAgent import YXAgent
from ParsAgent import ParsAgent
from caduceus import Caduceus
from caduceusDC16 import CaduceusDC16
from originalCaduceus import OriginalCaduceus
######################################################################
######################################################################

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20


class DenseLayer(nn.Module):
    def __init__(
            self,
            ensemble_size,
            input_dim,
            output_dim,
            activation=F.relu,
            hidden_init=ptu.fanin_init,
            bias_const=0.0,
    ):
        super(DenseLayer, self).__init__()
        self.ensemble_size = ensemble_size
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation = activation

        weights = []
        biases = []
        for _ in range(ensemble_size):
            weight = torch.zeros(input_dim, output_dim)
            hidden_init(weight)
            weights.append(weight.unsqueeze(0))
            bias = torch.zeros(1, output_dim).unsqueeze(0) + bias_const
            biases.append(bias)
        self.weights = torch.cat(weights)  # num_ensemble_size, input_dim, output_dim
        self.weights = torch.nn.Parameter(data=self.weights, requires_grad=True)
        self.biases = torch.cat(biases)  # num_ensemble_size, 1, output_dim
        self.biases = torch.nn.Parameter(data=self.biases, requires_grad=True)

    def forward(self, x):
        # x: (num_ensemble_size, batch_size, indim), weights: (num_ensemble_size, indim, outdim), biases: (num_ensemble_size, 1, outdim)
        x = torch.einsum("bij,bjk->bik", [x, self.weights]) + self.biases
        if self.activation is None:
            return x
        else:
            return self.activation(x)  


class ParallelMlp(nn.Module):
    def __init__(
            self,
            input_dim,
            hidden_dim,
            num_hidden_layers,
            output_dim,
            ensemble_size,
            bias_const=0.0,
            init_w=3e-3,
    ):
        super(ParallelMlp, self).__init__()

        self.ensemble_size = ensemble_size
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_hidden_layers = num_hidden_layers
        self.init_w = init_w

        self.h1 = DenseLayer(
            ensemble_size,
            input_dim=input_dim,
            output_dim=hidden_dim,
            hidden_init=ptu.fanin_init,
        )
        self.h2 = DenseLayer(
            ensemble_size,
            input_dim=hidden_dim,
            output_dim=hidden_dim,
            hidden_init=ptu.fanin_init,
        )
        if self.num_hidden_layers == 3:
            self.h3 = DenseLayer(
                ensemble_size,
                input_dim=hidden_dim,
                output_dim=hidden_dim,
                hidden_init=ptu.fanin_init,
            )
        self.output = DenseLayer(
            ensemble_size,
            input_dim=hidden_dim,
            output_dim=output_dim,
            activation=None,
            hidden_init=self.last_fc_init,
            bias_const=bias_const,
        )

    def last_fc_init(self, tensor):
        tensor.data.uniform_(-self.init_w, self.init_w)

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=-1)
        x = self.h1(x)
        x = self.h2(x)
        if self.num_hidden_layers == 3:
            x = self.h3(x)
        x = self.output(x)
        return x

    def weight_norm(self):
        return (
                torch.norm(self.h1.weights)
                + torch.norm(self.h2.weights)
                + torch.norm(self.output.weights)
        )


######################################################################
######################################################################

class ParallelTanhGaussianPolicy(nn.Module):
    def __init__(self, policies):
        super(ParallelTanhGaussianPolicy, self).__init__()
        self.policies = nn.ModuleList(policies)
    def get_policies(self):
        return self.policies
    def forward(self, obs):
        mean_list, std_list = [], []
        for policy in self.policies:
            tanh_normal = policy(obs)
            mean = tanh_normal.normal_mean.unsqueeze(0) 
            std = tanh_normal.normal_std.unsqueeze(0)  
            mean_list.append(mean)
            std_list.append(std)

        means = torch.cat(mean_list)
        stds = torch.cat(std_list)

        avg_mean = means.mean(0).unsqueeze(0)  
        avg_var = (means ** 2 + stds ** 2).mean(0).unsqueeze(0) - avg_mean ** 2
        avg_std = avg_var.sqrt()

        avg_mean = avg_mean.squeeze(0)
        avg_std = avg_std.squeeze(0)
        avg_std = torch.clamp(avg_std, np.exp(LOG_SIG_MIN), np.exp(LOG_SIG_MAX))

        return TanhNormal(avg_mean, avg_std)

    def logprob(self, action, mean, std):
        tanh_normal = TanhNormal(mean, std)
        log_prob = tanh_normal.log_prob(
            action,
        )
        log_prob = log_prob.sum(dim=1, keepdim=True)
        return log_prob

    def get_action(
            self,
            obs_np,
    ):
        actions = self.get_actions(obs_np[None])
        return actions[0, :], {}

    def get_actions(
            self,
            obs_np,
    ):
        dist = self._get_dist_from_np(obs_np)
        actions = dist.sample()
        return elem_or_tuple_to_numpy(actions)

    def _get_dist_from_np(self, *args, **kwargs):
        torch_args = tuple(torch_ify(x) for x in args)
        torch_kwargs = {k: torch_ify(v) for k, v in kwargs.items()}
        dist = self.forward(*torch_args, **torch_kwargs)
        return dist

    def reset(self):
        pass


def experiment(variant, args):

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    ################
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_no
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # RL agent to be trained
    
    # if args.offer_model != "":
    #     rl_agent.load(args.offer_model, args.seed)
    # opponent
    ponpoko_agent = PonPokoAgent(max_round=args.max_round, name="ponpoko agent")
    # parscat_agent = ParsCat(max_round=args.max_round, name="parscat agent")
    atlas3_agent = Atlas3(max_round=args.max_round, name="atlas3 agent")
    agentlg_agent = AgentLG(max_round=args.max_round, name="AgentLG agent")
    omac_agent = OMAC(max_round=args.max_round, name="omac agent")
    CUHK_agent = CUHKAgent(max_round=args.max_round, name="CUHK agent")
    HardHeaded_agent = HardHeadedAgent(max_round=args.max_round, name="HardHeaded agent")
    YX_Agent = YXAgent(max_round=args.max_round, name="YXAgent")
    ParsAgent_agent = ParsAgent(max_round=args.max_round, name="ParsAgent")
    AgreeableAgent2018_agent=AgreeableAgent2018(max_round=args.max_round, name="AgreeableAgent2018")
    caduceus_agent = Caduceus(max_round=args.max_round, name="Caduceus agent")
    opponents_pool = []
    opponents_pool.append(ponpoko_agent)
    opponents_pool.append(atlas3_agent)
    opponents_pool.append(agentlg_agent)
    opponents_pool.append(omac_agent)
    opponents_pool.append(CUHK_agent)
    opponents_pool.append(YX_Agent)
    opponent=ponpoko_agent
    obs_dim = variant['state_dim']
    action_dim = variant['action_dim']

    M = variant["layer_size"]
    num_hidden_layers = 2

    """ Prepare networks """
    weight_net = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M] * num_hidden_layers,
    )
    weight_net.last_fc.bias.data.fill_(1.0)

    weight_net.to(ptu.device)

    """ Prepare networks """
    qf1 = ParallelMlp(
        obs_dim + action_dim, M, num_hidden_layers, 1, args.ensemble_size
    ).to(ptu.device)

    qf2 = ParallelMlp(
        obs_dim + action_dim, M, num_hidden_layers, 1, args.ensemble_size
    ).to(ptu.device)

    target_qf1 = ParallelMlp(
        obs_dim + action_dim, M, num_hidden_layers, 1, args.ensemble_size
    ).to(ptu.device)

    target_qf2 = ParallelMlp(
        obs_dim + action_dim, M, num_hidden_layers, 1, args.ensemble_size
    ).to(ptu.device)
    
    policies = []
    for i in range(args.ensemble_size):
        ens_policy = TanhGaussianPolicy(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_sizes=[M] * num_hidden_layers,
        ).to(ptu.device)

        """ Load the networks """
        
        saved_model_dir = (
                "./data/cql/cql_models_" + args.domain + "_seed{}".format(i * 5 + args.seed))
        

        qf1_params = torch.load('%s/iter500000_critic1.pt'%saved_model_dir)
        qf2_params = torch.load('%s/iter500000_critic2.pt'%saved_model_dir)
        #print("qf2_params：", qf2_params)
        with torch.no_grad():
            qf1.h1.weights[i] = qf1_params["h1.weight"]
            qf1.h1.biases[i][0] = qf1_params["h1.bias"]
            qf1.h2.weights[i] = qf1_params["h2.weight"]
            qf1.h2.biases[i][0] = qf1_params["h2.bias"]
            qf1.output.weights[i] = qf1_params["output.weight"]
            qf1.output.biases[i][0] = qf1_params["output.bias"]
            qf2.h1.weights[i] = qf2_params["h1.weight"]
            qf2.h1.biases[i][0] = qf2_params["h1.bias"]
            qf2.h2.weights[i] = qf2_params["h2.weight"]
            qf2.h2.biases[i][0] = qf2_params["h2.bias"]
            qf2.output.weights[i] = qf2_params["output.weight"]
            qf2.output.biases[i][0] = qf2_params["output.bias"]

        ens_policy.load_state_dict(
            torch.load('%s/iter500000_actor.pt'%saved_model_dir)
        )
        policies.append(ens_policy)

    policy = ParallelTanhGaussianPolicy(policies).to(ptu.device)

    target_qf1.load_state_dict(qf1.state_dict())
    target_qf2.load_state_dict(qf2.state_dict())
    rl_agent = RLAgent(args.max_round, 'rl agent', device,policy,qf1,qf2,target_qf1,target_qf2, 3, variant['algorithm'])
    eval_policy = MakeDeterministic(policy)
    eval_path_collector = MdpPathCollector(
        max_round=args.max_round,
        domain_type=args.domain_type,#DISCRETE
        domain=args.domain,#Amsterdam
        rl_agent=rl_agent,
        opponent=opponent,
        opponents_pool=opponents_pool,
        start_timesteps=0,# 5e4
        policy=eval_policy,
        render=True
    )
    expl_path_collector = MdpPathCollector(
        max_round=args.max_round,  
        domain_type=args.domain_type,  # DISCRETE
        domain=args.domain,  # Amsterdam
        rl_agent=rl_agent,
        opponent=opponent,
        opponents_pool=opponents_pool,
        start_timesteps=0,  # 5e4
        policy=policy,
        render=False
    )
    buffer_filename = None  
    if variant['buffer_filename'] is not None:
        buffer_filename = variant['buffer_filename']



    dataset=np.load(buffer_filename,allow_pickle=True)
    dataset_size=dataset.shape[0]


    priority_replay_buffer = PriorityReplayBuffer(
        int(2 ** np.ceil(np.log2(dataset_size + args.online_buffer_size))),
        action_dim=action_dim,
        observation_dim=obs_dim
    )

    offline_replay_buffer = EnvReplayBuffer(
        variant["replay_buffer_size"],
        action_dim = action_dim,
        observation_dim = obs_dim
        
    )

    online_replay_buffer = EnvReplayBuffer(
        args.online_buffer_size,
        action_dim=action_dim,
        observation_dim=obs_dim
       
    )


    for i in range(dataset_size - 1):
        obs = dataset[i][0]
        new_obs = dataset[i][3]
        action =dataset[i][1]
        reward = dataset[i][2]
        done = dataset[i][4]


        priority_replay_buffer.add_sample(
            observation=obs,
            action=action,
            reward=reward,
            next_observation=new_obs,
            terminal=done,
            Discrete=False
        )
        offline_replay_buffer.add_sample(
            observation=obs,
            action=action,
            reward=reward,
            next_observation=new_obs,
            terminal=done,
            Discrete=False
        )
    print("load replay buffer")
    trainer = OursTrainer(

        rl_agent=rl_agent,
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        alpha_lr=args.policy_lr,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        weight_net=weight_net,
        save_model_dir=args.save_model_dir,
        weight_net_lr=args.weight_net_lr,
        priority_replay_buffer=priority_replay_buffer,
        temperature=args.temperature,
        offline_dataset_size=offline_replay_buffer._size,  # for logging
        ensemble_size=args.ensemble_size,
        **variant["trainer_kwargs"]
    )

    algorithm = TorchOurRLAlgorithm(
        trainer=trainer,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        offline_replay_buffer=offline_replay_buffer,
        online_replay_buffer=online_replay_buffer,
        priority_replay_buffer=priority_replay_buffer,
        first_epoch_multiplier=args.first_epoch_multiplier,
        weight_net_batch_size=args.weight_net_batch_size,
        init_online_fraction=args.init_online_fraction,
        **variant["algorithm_kwargs"]
    )

    algorithm.to(ptu.device)
    algorithm.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--domain", default="HouseKeeping", type=str)
    parser.add_argument("--domain_type", default="DISCRETE", type=str)
    parser.add_argument("--min_steps_before_training", default=0, type=int)
    parser.add_argument("--max_path_length", default=1000, type=int)

    # RL
    parser.add_argument("--policy_lr", default=3e-4, type=float)
    parser.add_argument("--lr", default=3e-4, type=float)
    parser.add_argument("--ensemble_size", default=5, type=int)

    # Fine-tuning hyperparameters
    parser.add_argument("--first_epoch_multiplier", default=5, type=int)#10
    parser.add_argument("--num_epochs", default=300, type=int)#251

    # Buffer flags
    parser.add_argument("--init_online_fraction", default=0.5, type=float)

    # Weight net hyperparameters
    parser.add_argument("--temperature", default=5.0, type=float)
    parser.add_argument("--weight_net_batch_size", default=256, type=int)
    parser.add_argument("--weight_net_lr", default=3e-4, type=float)
    parser.add_argument("--online_buffer_size", default=250000, type=int)
    parser.add_argument("--buffer_filename", default="/home/ubuntu/multi_issue_negotiation/laptop.npy", type=str)
    #nego
    parser.add_argument("--start_timesteps", default=0, type=int)
    parser.add_argument("--max_round", default=1000, type=int)
    parser.add_argument("--gpu_no", default='0', type=str)
    parser.add_argument("--oppo_type", default='Atlas3', type=str)

    parser.add_argument('--save_models', default=1, type=int)
    parser.add_argument('--save_model_dir', default="", type=str)
    parser.add_argument('--save_every_step', default=10000, type=int)

    args = parser.parse_args()

    log_dir = (
            "ours/"
            + args.domain
    )

    variant = dict(
        algorithm="SAC",
        version="normal",
        layer_size=256,
        replay_buffer_size=int(2.5e6),
        buffer_filename="",
        state_dim=7,
        action_dim=1,
        algorithm_kwargs=dict(
            num_epochs=args.num_epochs,
            num_eval_steps_per_epoch=5000,
            num_trains_per_train_loop=1000,
            num_expl_steps_per_train_loop=1000,
            min_num_steps_before_training=args.min_steps_before_training,
            max_path_length=args.max_path_length,
            batch_size=256,

        ),
        trainer_kwargs=dict(
            discount=0.99,
            soft_target_tau=5e-3,
            target_update_period=1,
            policy_lr=args.policy_lr,
            qf_lr=args.lr,
            reward_scale=1,
            use_automatic_entropy_tuning=True,

        ),
    )
    args.save_model_dir = "data/fintune/ensemble_ponpoko_" + str(args.domain) + "_seed" + str(args.seed)
    if args.save_models:
        if variant['algorithm'] == 'SAC' and not os.path.exists(args.save_model_dir):
            os.makedirs(args.save_model_dir)
    variant['buffer_filename']=args.buffer_filename
    setup_logger(log_dir, variant=variant)
    ptu.set_gpu_mode(True)
    experiment(variant, args)
