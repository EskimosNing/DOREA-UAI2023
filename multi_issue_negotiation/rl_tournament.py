import numpy as np
import os
import argparse
import random
from utils import get_utility
from utils import ReplayBuffer
from rlagent import RLAgent
from agent import TimeAgentBoulware
from agent import BehaviorAgentAverage
from negotiation import Negotiation
import torch
from atlas3 import Atlas3
from ponpokoagent import PonPokoAgent
from parscat import ParsCat
from agentlg import AgentLG
from omacagent import OMAC
from CUHKAgent import CUHKAgent
from randomagent import RandomAgent
from HardHeaded import HardHeadedAgent
from YXAgent import YXAgent
from ParsAgent import ParsAgent
from caduceus import Caduceus
from AgreeableAgent2018 import AgreeableAgent2018
import pandas as pd



def evaluate_policy(negotiation, opposition="low", eval_episodes=500, render=False, rl_agent_group=None, agent_group_1=None, args=None):
    rows = []
    for v in agent_group_1:
        for u in rl_agent_group:        

            rl_agent = u
            opponent = v

            avg_reward = 0
            avg_round = 0
            avg_oppo = 0
            succ_avg_reward = 0            
            succ_avg_round = 0
            succ_avg_oppo = 0
            succ_counts = 0

            succ_domains = []
            row = []

            # [0, 0.25] 10, [0.25, 0.5] 6, [0.5, 1] 4 
            allDomains = ["Acquisition", "Animal", "Coffee", "DefensiveCharms", "Camera", "DogChoosing", "FiftyFifty2013", \
                        "HouseKeeping", "Icecream", "Kitchen", "Laptop", "NiceOrDie", "Outfit", "planes", "SmartPhone", \
                        "Ultimatum", "Wholesaler", "RentalHouse-B", "Barter-C", "Amsterdam-B"]

            for j in range(eval_episodes):
                if render:
                    print("----------- a new episode when evaluating ---------")

                if args.domain_type == "REAL":
                    negotiation = Negotiation(max_round=args.max_round, issue_num=3, domain_type=args.domain_type, domain_file=args.domain_file)
                elif args.domain_type == "DISCRETE":
                    negotiation = Negotiation(max_round=args.max_round, domain_type=args.domain_type, domain_file=args.domain_file)

                negotiation.clear()

                negotiation.add(rl_agent)   
                negotiation.add(opponent)   

                if render:
                    if negotiation.domain_type == 'DISCRETE':
                        print("DISCRETE domain : ", allDomains[j%len(allDomains)])
                negotiation.reset(opposition=opposition, domain_file=allDomains[j%len(allDomains)])

                last_offer = None
                accept = False
                current_player = 0
                episode_reward = 0
                episode_round = 0
                for i in range(1, negotiation.max_round + 1):
                    if render:
                        print("Round:", i)
                    current_player = 1 - i % 2

                    negotiation.agents_pool[current_player].set_t(i)
                    negotiation.agents_pool[1 - current_player].set_t(i)
                    if i == 1:
                        negotiation.agents_pool[current_player].receive(last_offer)
                    
                    if negotiation.agents_pool[current_player].__class__.__name__ == "RLAgent":
                        if render:
                            print("  RL agent's obs: ", rl_agent.obs)
                        action = negotiation.agents_pool[current_player].act()

                        last_utility = 0.5 * (action + 1) * (rl_agent.u_max - rl_agent.u_min) + rl_agent.u_min
                        rl_agent.s = last_utility
                        last_offer = rl_agent.gen_offer()
                    else:
                        last_offer = negotiation.agents_pool[current_player].act()
                    
                    if (last_offer is None) and (negotiation.agents_pool[current_player].__class__.__name__ == "CUHKAgent" or negotiation.agents_pool[current_player].__class__.__name__ == "HardHeadedAgent" \
                        or negotiation.agents_pool[current_player].__class__.__name__ == "YXAgent" or negotiation.agents_pool[current_player].__class__.__name__ == "OMAC" \
                        or negotiation.agents_pool[current_player].__class__.__name__ == "AgentLG" or negotiation.agents_pool[current_player].__class__.__name__ == "ParsAgent" or negotiation.agents_pool[current_player].__class__.__name__ == "Caduceus" or negotiation.agents_pool[current_player].__class__.__name__ == "Atlas3" or negotiation.agents_pool[current_player].__class__.__name__ == "PonPokoAgent" or negotiation.agents_pool[current_player].__class__.__name__ == "ParsCat"):
                        if negotiation.agents_pool[current_player].accept == True:                     
                            accept = True
                            episode_round = i
                            if render:
                                print(negotiation.agents_pool[current_player].name, "accept the offer.\n")                    
                            episode_reward = negotiation.agents_pool[1-current_player].utility_proposed[-1]
                            avg_oppo +=  negotiation.agents_pool[current_player].utility_received[-1]
                            succ_avg_reward += episode_reward
                            succ_avg_oppo += negotiation.agents_pool[current_player].utility_received[-1]
                            succ_counts += 1
                            succ_avg_round += episode_round
                            if allDomains[j%len(allDomains)] not in succ_domains:
                                succ_domains.append(allDomains[j%len(allDomains)])
                            break                
                        elif negotiation.agents_pool[current_player].terminate == True:  
                            if render: 
                                print(negotiation.agents_pool[current_player].name, "end the negotiation.")
                            episode_reward = 0
                        break
                    elif last_offer is None:
                        print("Error exist: agent's offer is None.")
                        exit(-1)
                    negotiation.agents_pool[1 - current_player].receive(last_offer)
                    if render:
                        print("  " + negotiation.agents_pool[current_player].name, "'s action", last_offer)
                        print("  utility to %s: %f, utility to %s: %f\n" % (negotiation.agents_pool[current_player].name,
                            get_utility(last_offer, negotiation.agents_pool[current_player].prefer, negotiation.agents_pool[current_player].condition, negotiation.agents_pool[current_player].domain_type, negotiation.agents_pool[current_player].issue_value),
                            negotiation.agents_pool[1 - current_player].name, get_utility(last_offer, negotiation.agents_pool[1 - current_player].prefer,
                            negotiation.agents_pool[1 - current_player].condition, negotiation.agents_pool[1 - current_player].domain_type, negotiation.agents_pool[1 - current_player].issue_value)))
                    if i == negotiation.max_round:
                        episode_reward = 0
                    if negotiation.agents_pool[1 - current_player].accept:
                        accept = True
                        episode_round = i + 1
                        if render:
                            print("Round:", i+1)
                            print("  "+negotiation.agents_pool[1 - current_player].name, "accept the offer.\n")
                        if last_offer is None:
                            episode_reward = 0
                        else:
                            episode_reward = get_utility(offer=last_offer, prefer=rl_agent.prefer, condition=rl_agent.condition, domain_type=rl_agent.domain_type, issue_value=rl_agent.issue_value)
                            avg_oppo +=  get_utility(offer=last_offer, prefer=opponent.prefer, condition=opponent.condition, domain_type=opponent.domain_type, issue_value=opponent.issue_value)
                            succ_avg_reward += episode_reward
                            succ_avg_oppo += get_utility(offer=last_offer, prefer=opponent.prefer, condition=opponent.condition, domain_type=opponent.domain_type, issue_value=opponent.issue_value)
                            succ_counts += 1
                            succ_avg_round += episode_round
                            if allDomains[j%len(allDomains)] not in succ_domains:
                                succ_domains.append(allDomains[j%len(allDomains)])
                        break

                    if render:
                        print()

                if render:
                    if accept:
                        print("Negotiation success")
                    else:                
                        print("Negotiation failed")
                    print("rl received reward: %f\n" % episode_reward)
                
                if accept == False:
                    episode_round = args.max_round

                avg_reward += episode_reward
                avg_round += episode_round
                
            avg_reward /= eval_episodes
            avg_round /= eval_episodes
            avg_oppo /= eval_episodes

            if succ_counts != 0:
                succ_avg_reward /= succ_counts                
                succ_avg_round /= succ_counts
                succ_avg_oppo /= succ_counts

            
            row.append(opponent.__class__.__name__)
            row.append(rl_agent.name)
            row.append(succ_avg_oppo)
            row.append(succ_avg_reward)
            row.append(avg_oppo)
            row.append(avg_reward)
            row.append(succ_avg_round)
            row.append(avg_round)
            row.append(succ_counts)
            row.append(eval_episodes)

            rows.append(row)

    df1 = pd.DataFrame(rows, columns=['Agent1', 'Agent2', 'utility1_succ', 'utility2_succ', 'utility1_total', 'utility2_total', 'avg_round_succ', 'avg_round_total', 'num_succ',	'num_total'])
    df1.to_excel("best_RL_against_ANAC.xlsx")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=9, type=int)                         # Sets PyTorch and Numpy seeds
    parser.add_argument("--max_round", default=30, type=int)                 # How many steps in an negotiation
    parser.add_argument("--save_models", action="store_true", default=True)    # Whether or not models are saved
    parser.add_argument("--expl_noise", default=0.1, type=float)               # Std of Gaussian exploration noise
    parser.add_argument("--batch_size", default=128, type=int)                 # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99, type=float)                # Discount factor
    parser.add_argument("--tau", default=0.002, type=float)                    # Target network update rate
    parser.add_argument("--use_automatic_entropy_tuning", default=True, type=bool)
    parser.add_argument("--target_entropy", default=None, type=float)
    parser.add_argument("--gpu_no", default='0', type=str)                    # GPU number, -1 means CPU
    parser.add_argument("--eval_opposition" ,default="low", type=str)
    parser.add_argument("--domain_type", default="DISCRETE", type=str)             # "REAL" or "DISCRETE"
    parser.add_argument("--domain_file", default="Acquisition", type=str)     # Only the DISCRETE domain needs to specify this arg 
    args = parser.parse_args()

    if args.save_models and not os.path.exists("./sac_models"):
        os.makedirs("./sac_models")

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_no
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.domain_type == "REAL":
        negotiation = Negotiation(max_round=args.max_round, issue_num=3, domain_type=args.domain_type, domain_file=args.domain_file)
    elif args.domain_type == "DISCRETE":
        negotiation = Negotiation(max_round=args.max_round, domain_type=args.domain_type, domain_file=args.domain_file)
    
    time_agent = TimeAgentBoulware(max_round=args.max_round, name="TimeAgentBoulware")
    behavior_agent = BehaviorAgentAverage(max_round=args.max_round, name="BehaviorAgentAverage", kind_rate=1)
    rl_agent = RLAgent(max_round=args.max_round, name="rl agent", device=device, use_automatic_entropy_tuning=args.use_automatic_entropy_tuning, target_entropy=args.target_entropy)
    ponpoko_agent = PonPokoAgent(max_round=args.max_round, name="ponpoko agent")
    parscat_agent = ParsCat(max_round=args.max_round, name="parscat agent")
    atlas3_agent = Atlas3(max_round=args.max_round, name="atlas3 agent")
    agentlg_agent = AgentLG(max_round=args.max_round, name="AgentLG agent")
    omac_agent = OMAC(max_round=args.max_round, name="omac agent")
    CUHK_agent = CUHKAgent(max_round=args.max_round, name="CUHK agent")
    random_agent = RandomAgent(max_round=args.max_round, name="random agent")
    HardHeaded_agent = HardHeadedAgent(max_round=args.max_round, name="HardHeaded agent")
    YX_Agent = YXAgent(max_round=args.max_round, name="YXAgent")
    ParsAgent_agent = ParsAgent(max_round=args.max_round, name="ParsAgent")
    caduceus_agent = Caduceus(max_round=args.max_round, name="Caduceus agent")


    rl_agent_group = []

    offer_model_list = ["ponpoko", "yxagent", 'parscat', 'atlas3', 'parsagent', 'agentlg', 'cuhkagent', 'hardheaded', 'omac', 'caduceus']
    
    for i in range(len(offer_model_list)):
        rl_agent = RLAgent(max_round=args.max_round, name="rl agent", device=device, use_automatic_entropy_tuning=args.use_automatic_entropy_tuning, target_entropy=args.target_entropy)
        rl_agent.load(offer_model_name=offer_model_list[i])
        rl_agent.name = "RL(against " + offer_model_list[i] + ")"
        rl_agent_group.append(rl_agent)


    agent_group_1 = [ponpoko_agent, parscat_agent, atlas3_agent, agentlg_agent, YX_Agent, CUHK_agent, ParsAgent_agent, HardHeaded_agent, omac_agent, caduceus_agent]


    print("start testing ...")
    evaluate_policy(negotiation, opposition=args.eval_opposition, eval_episodes=200, render=False, rl_agent_group=rl_agent_group, agent_group_1=agent_group_1, args=args)
