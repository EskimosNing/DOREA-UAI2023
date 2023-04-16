import rlkit.torch.pytorch_util as ptu
from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
from rlkit.launchers.launcher_util import setup_logger
from rlkit.samplers.data_collector import MdpPathCollector, CustomMDPPathCollector
from rlkit.torch.sac.policies import TanhGaussianPolicy, MakeDeterministic
from algorithm.cql import CQLTrainer
from rlkit.torch.torch_rl_algorithm import TorchOnlineRLAlgorithm
import torch.nn.functional as F
from rlkit.torch.networks.mlp import ConcatMlp
from rlkit.torch.distributions import TanhNormal
import torch
import argparse, os
import numpy as np
import torch.nn as nn
import random
from algorithm.sac import SACTrainer

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


LOG_SIG_MAX = 2
LOG_SIG_MIN = -20


class DenseLayer(nn.Module):
    def __init__(
            self,
            input_dim,
            output_dim,
            activation=F.relu,
            hidden_init=ptu.fanin_init,
            bias_const=0.0,
    ):
        super(DenseLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation = activation
        self.weight = torch.zeros(input_dim, output_dim)
        hidden_init(self.weight)
        self.bias = torch.zeros(1, output_dim) + bias_const
        self.weight = torch.nn.Parameter(data=self.weight, requires_grad=True)

        self.bias = torch.nn.Parameter(data=self.bias, requires_grad=True)

    def forward(self, x):
        x = torch.einsum("ij,jk->ik", [x, self.weight]) + self.bias
        if self.activation is None:
            return x
        else:
            return self.activation(x)  


class SingleMlp(nn.Module):
    def __init__(
            self,
            input_dim,
            hidden_dim,
            num_hidden_layers,
            output_dim,

            bias_const=0.0,
            init_w=3e-3,
    ):  
        super(SingleMlp, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_hidden_layers = num_hidden_layers
        self.init_w = init_w

        self.h1 = DenseLayer(
            input_dim=input_dim,
            output_dim=hidden_dim,
            hidden_init=ptu.fanin_init,
        )  
        self.h2 = DenseLayer(
            input_dim=hidden_dim,
            output_dim=hidden_dim,
            hidden_init=ptu.fanin_init,
        )  
        if self.num_hidden_layers == 3:
            self.h3 = DenseLayer(

                input_dim=hidden_dim,
                output_dim=hidden_dim,
                hidden_init=ptu.fanin_init,
            )  
        self.output = DenseLayer(

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


def experiment(variant, args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_no
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    obs_dim = args.state_dim
    action_dim = args.action_dim
    num_hidden_layers = 2  

    """ Prepare networks """
    M = variant['layer_size']


    """ Prepare networks """

    qf1 = SingleMlp(
        obs_dim + action_dim, M, num_hidden_layers, 1
    ).to(ptu.device)  # obs_dim + action_dim, M, num_hidden_layers, 1, args.ensemble_size
    qf2 = SingleMlp(
        obs_dim + action_dim, M, num_hidden_layers, 1
    ).to(ptu.device)
    target_qf1 = SingleMlp(
        obs_dim + action_dim, M, num_hidden_layers, 1
    ).to(ptu.device)
    target_qf2 = SingleMlp(
        obs_dim + action_dim, M, num_hidden_layers, 1
    ).to(ptu.device)

    policy = TanhGaussianPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_sizes=[M] * num_hidden_layers,
    ).to(ptu.device) 

    ponpoko_agent = PonPokoAgent(max_round=args.max_round, name="ponpoko agent")
    parscat_agent = ParsCat(max_round=args.max_round, name="parscat agent")
    atlas3_agent = Atlas3(max_round=args.max_round, name="atlas3 agent")
    agentlg_agent = AgentLG(max_round=args.max_round, name="AgentLG agent")
    omac_agent = OMAC(max_round=args.max_round, name="omac agent")
    CUHK_agent = CUHKAgent(max_round=args.max_round, name="CUHK agent")
    HardHeaded_agent = HardHeadedAgent(max_round=args.max_round, name="HardHeaded agent")
    YX_Agent = YXAgent(max_round=args.max_round, name="YXAgent")
    ParsAgent_agent = ParsAgent(max_round=args.max_round, name="ParsAgent")
    caduceus_agent = Caduceus(max_round=args.max_round, name="Caduceus agent")

    opponent = ponpoko_agent
    opponents_pool=[]
    opponents_pool.append(ponpoko_agent)
    opponents_pool.append(atlas3_agent)
    opponents_pool.append(agentlg_agent)
    opponents_pool.append(omac_agent)
    opponents_pool.append(CUHK_agent)
    opponents_pool.append(YX_Agent)

    rl_agent = RLAgent(args.max_round, 'rl agent', device, policy, qf1, qf2, target_qf1, target_qf2, 3,variant['algorithm'])


    eval_policy = MakeDeterministic(policy)
    eval_path_collector = MdpPathCollector(

        max_round=args.max_round,
        domain_type=args.domain_type,
        domain=args.domain,
        rl_agent=rl_agent,
        opponent=opponent,
        opponents_pool=opponents_pool,
        start_timesteps= 5e4,
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

    replay_buffer = EnvReplayBuffer(variant['replay_buffer_size'], action_dim=args.action_dim,
                                    observation_dim=args.state_dim)  

    if variant['load_buffer'] and buffer_filename is not None:
        replay_buffer.load_buffer(buffer_filename)

    ''' 
    #replay——buffer
    buffer_filename = None
    if variant['buffer_filename'] is not None:
        buffer_filename = variant['buffer_filename']

    replay_buffer = EnvReplayBuffer(
        variant['replay_buffer_size'],
        expl_env,
    )

    if variant['load_buffer'] and buffer_filename is not None:
        replay_buffer.load_buffer(buffer_filename)
    elif 'random-expert' in variant['env_name']:
        load_hdf5(d4rl.basic_dataset(eval_env), replay_buffer) 
    else:
        load_hdf5(d4rl.qlearning_dataset(eval_env), replay_buffer)
    '''

    trainer = SACTrainer(
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        save_model_dir=args.save_model_dir,
        **variant['trainer_kwargs']
    )
    algorithm = TorchOnlineRLAlgorithm(
        trainer=trainer,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,

        **variant['algorithm_kwargs']
    )
    algorithm.to(ptu.device)
    algorithm.train()


def enable_gpus(gpu_str):
    if (gpu_str is not ""):
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_str
    return


if __name__ == "__main__":
    variant = dict(
        algorithm="SAC",
        version="normal",
        domain="HouseKeeping",
        state_dim=7,
        action_dim=1,
        layer_size=256,
        replay_buffer_size=int(2E6),
        buffer_filename=None,
        load_buffer=None,


        sparse_reward=False,
        algorithm_kwargs=dict(
            num_epochs=500,  
            num_eval_steps_per_epoch=1000,
            num_trains_per_train_loop=1000,
            num_expl_steps_per_train_loop=1000,
            min_num_steps_before_training=1000,
            max_path_length=1000,
            batch_size=256,
        ),
        trainer_kwargs=dict(

            discount=0.99,
            soft_target_tau=5e-3,
            policy_lr=1E-4,
            qf_lr=3E-4,
            reward_scale=1,
            use_automatic_entropy_tuning=True,

            policy_eval_start=40000,


            state_dim=7,
            action_dim=1,
            save_models=1,
            save_every_step=1000,

        ),
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", type=str, default='HouseKeeping')
    parser.add_argument("--state_dim", type=int, default='7')
    parser.add_argument("--action_dim", type=int, default='1')
    parser.add_argument("--gpu", default='0', type=str)
    parser.add_argument("--domain_type", default="DISCRETE", type=str)
    parser.add_argument("--policy_eval_start", default=40000,type=int)  # Defaulted to 20000 (40000 or 10000 work similarly)
    parser.add_argument('--policy_lr', default=1e-4, type=float)  # Policy learning rate

    parser.add_argument('--seed', default=0, type=int)  # 10，0，5,15
    parser.add_argument('--save_models', default=1, type=int)
    parser.add_argument('--save_model_dir', default="", type=str)
    parser.add_argument('--save_every_step', default=10000, type=int)

    # nego
    parser.add_argument("--start_timesteps", default=0, type=int)
    parser.add_argument("--max_round", default=300, type=int)
    parser.add_argument("--gpu_no", default='0', type=str)
    parser.add_argument("--oppo_type", default='ponpoko', type=str)

    args = parser.parse_args()
    enable_gpus(args.gpu)

    variant['trainer_kwargs']['policy_lr'] = args.policy_lr

    variant['trainer_kwargs']['policy_eval_start'] = args.policy_eval_start

    variant['trainer_kwargs']['state_dim'] = args.state_dim
    variant['trainer_kwargs']['action_dim'] = args.action_dim
    variant['trainer_kwargs']['save_models'] = args.save_models
    variant['trainer_kwargs']['save_every_step'] = args.save_every_step

    variant['buffer_filename'] = None

    variant['load_buffer'] = False

    variant['domain'] = args.domain
    variant['state_dim'] = args.state_dim
    variant['action_dim'] = args.action_dim
    variant['seed'] = args.seed
    args.save_model_dir = "data/sac/sac_ponpoko_" + str(args.domain) + "_seed" + str(args.seed)

    if args.save_models:
        if variant['algorithm'] == 'SAC' and not os.path.exists(args.save_model_dir):
            os.makedirs(args.save_model_dir)

    ptu.set_gpu_mode(True)
    
    experiment(variant, args)
