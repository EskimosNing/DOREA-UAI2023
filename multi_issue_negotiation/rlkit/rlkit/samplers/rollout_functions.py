import math
from functools import partial

import numpy as np
import copy

import torch

from utils import  get_utility_with_discount
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

create_rollout_function = partial


def multitask_rollout(
        env,
        agent,
        max_path_length=np.inf,
        render=False,
        render_kwargs=None,
        observation_key=None,
        desired_goal_key=None,
        get_action_kwargs=None,
        return_dict_obs=False,
        full_o_postprocess_func=None,
):
    if full_o_postprocess_func:
        def wrapped_fun(env, agent, o):
            full_o_postprocess_func(env, agent, observation_key, o)
    else:
        wrapped_fun = None

    def obs_processor(o):
        return np.hstack((o[observation_key], o[desired_goal_key]))

    paths = rollout(
        env,
        agent,
        max_path_length=max_path_length,
        render=render,
        render_kwargs=render_kwargs,
        get_action_kwargs=get_action_kwargs,
        preprocess_obs_for_policy_fn=obs_processor,
        full_o_postprocess_func=wrapped_fun,
    )
    if not return_dict_obs:
        paths['observations'] = paths['observations'][observation_key]
    return paths


def contextual_rollout(
        env,
        agent,
        observation_key=None,
        context_keys_for_policy=None,
        obs_processor=None,
        **kwargs
):
    if context_keys_for_policy is None:
        context_keys_for_policy = ['context']

    if not obs_processor:
        def obs_processor(o):
            combined_obs = [o[observation_key]]
            for k in context_keys_for_policy:
                combined_obs.append(o[k])
            return np.concatenate(combined_obs, axis=0)
    paths = rollout(
        env,
        agent,
        preprocess_obs_for_policy_fn=obs_processor,
        **kwargs
    )
    return paths


def rollout(
        negotiation,
        agent,

        rl_agent,
        opponent,
        start_timesteps,
        max_path_length=np.inf,
        opponents_pool=None,
        render=False,
        render_kwargs=None,
        preprocess_obs_for_policy_fn=None,
        get_action_kwargs=None,
        return_dict_obs=False,
        full_o_postprocess_func=None,
        reset_callback=None,
):

    observations = []
    actions = []
    rewards = []
    terminals = []
    max_path_length1=max_path_length
    next_observations = []
    path_length = 0
    #agent.reset()#

    total_timesteps = 0
    timesteps_since_eval = 0
    episode_num = 0
    episode_num_last_eval = 0
    #opponent=PonPokoAgent(max_round=1500, name="ponpoko agent")
    negotiation.add(opponent)
    negotiation.add(rl_agent)
    #negotiation.add(opponent)
    negotiation.reset()

    episode_reward = 0
    episode_timesteps = 0
    episode_num += 1
    episode_num_last_eval += 1

    last_offer = None
    current_player = 0
    obs = None
    action = None
    new_obs = None


    avg_reward = 0
    avg_round = 0
    avg_oppo = 0
    succ_avg_reward = 0
    succ_counts = 0
    succ_avg_oppo = 0

    succ_domains = []

    eval_episodes=20

    if render:
        for j in range(eval_episodes):
            negotiation.reset()
            print("----------- a new episode when evaluating ---------")
            if negotiation.domain_type == 'DISCRETE':
                print("DISCRETE domain: ", negotiation.domain_file)
            last_offer = None
            accept = False
            current_player = 1
            episode_reward = 0
            episode_round = 0
            for i in range(1, negotiation.max_round + 1):
                print("Round:", i)
                #current_player=1-current_player
                end=0
                while end!=1:
                    
                    current_player = 1-current_player

                    #print(ne)
                    negotiation.agents_pool[current_player].set_t(i)
                    negotiation.agents_pool[1 - current_player].set_t(i)
                    if i == 1:
                        negotiation.agents_pool[current_player].receive(last_offer)
                    if negotiation.agents_pool[current_player].__class__.__name__ == "RLAgent":
                        print("  RL agent's obs: ", rl_agent.obs)
                        action = negotiation.agents_pool[current_player].act()#rl  ut
                        # last_utility=action * math.pow(negotiation.agents_pool[current_player].discount,
                        #                                negotiation.agents_pool[current_player].relative_t)
                        last_utility = 0.5 * (action + 1) * (rl_agent.u_max - rl_agent.u_min) + rl_agent.u_min
                        #last_utility = torch.clamp(action,rl_agent.u_min,rl_agent.u_max)
                        #last_utility = last_utility * math.pow(negotiation.agents_pool[current_player].discount,negotiation.agents_pool[current_player].relative_t)
                        rl_agent.s = last_utility
                        last_offer = rl_agent.gen_offer()
                    else:
                        last_offer = negotiation.agents_pool[current_player].act()
                    if (last_offer is None) and (
                            negotiation.agents_pool[current_player].__class__.__name__ == "CUHKAgent" or negotiation.agents_pool[
                        current_player].__class__.__name__ == "HardHeadedAgent" or negotiation.agents_pool[current_player].__class__.__name__ == "YXAgent" or
                            negotiation.agents_pool[current_player].__class__.__name__ == "OMAC" or
                            negotiation.agents_pool[current_player].__class__.__name__ == "AgentLG" or
                            negotiation.agents_pool[current_player].__class__.__name__ == "ParsAgent" or
                            negotiation.agents_pool[current_player].__class__.__name__ == "Caduceus" or
                            negotiation.agents_pool[current_player].__class__.__name__ == "Atlas3" or
                            negotiation.agents_pool[current_player].__class__.__name__ == "PonPokoAgent" or
                            negotiation.agents_pool[current_player].__class__.__name__ == "ParsCat" or
                            negotiation.agents_pool[current_player].__class__.__name__ == "AgreeableAgent2018" or
                            negotiation.agents_pool[current_player].__class__.__name__ == "Agent36" ):
                        if negotiation.agents_pool[current_player].accept == True:
                            accept = True
                            episode_round = i

                            print(negotiation.agents_pool[current_player].name, "accept the offer.\n")
                            episode_reward = negotiation.agents_pool[1 - current_player].utility_proposed[-1]
                            avg_oppo += negotiation.agents_pool[current_player].utility_received[-1]
                            succ_avg_reward += episode_reward
                            succ_avg_oppo += negotiation.agents_pool[current_player].utility_received[-1]
                            succ_counts += 1
                            if negotiation.domain_file not in succ_domains:
                                succ_domains.append(negotiation.domain_file)
                            break
                        elif negotiation.agents_pool[current_player].terminate == True:

                            print(negotiation.agents_pool[current_player].name, "end the negotiation.")
                            episode_reward = 0
                        break
                    elif last_offer is None:
                        print("Error exist: agent's offer is None.")
                        exit(-1)

                    negotiation.agents_pool[1 - current_player].receive(last_offer)

                    print("  " + negotiation.agents_pool[current_player].name, "'s action", last_offer)
                    #if negotiation.agents_pool[current_player].relative_t <1.0 and negotiation.agents_pool[1-current_player].relative_t <1.0:
                    print("  utility to %s: %f, utility to %s: %f\n" % (negotiation.agents_pool[current_player].name,
                                                                            get_utility(last_offer, negotiation.agents_pool[current_player].prefer,
                                                                            negotiation.agents_pool[current_player].condition,
                                                                            negotiation.agents_pool[current_player].domain_type,
                                                                            negotiation.agents_pool[current_player].issue_value
                                                                            
                                                                            ),
                                                                            negotiation.agents_pool[1 - current_player].name,
                                                                            get_utility(last_offer, negotiation.agents_pool[1 - current_player].prefer,
                                                                            negotiation.agents_pool[1 - current_player].condition,
                                                                            negotiation.agents_pool[1 - current_player].domain_type,
                                                                            negotiation.agents_pool[1 - current_player].issue_value
                                                                    
                                                                                        )))

                    if i == negotiation.max_round:
                        episode_reward = 0
                    if negotiation.agents_pool[1 - current_player].accept:
                        accept = True
                        episode_round = i + 1
                        if render:
                            print("Round:", i + 1)
                            print("  " + negotiation.agents_pool[1 - current_player].name, "accept the offer.\n")
                        if last_offer is None:
                            episode_reward = 0
                        else:
                            episode_reward = get_utility(offer=last_offer, prefer=rl_agent.prefer, condition=rl_agent.condition,
                                                         domain_type=rl_agent.domain_type, issue_value=rl_agent.issue_value)#self.relative_t,self.discount
                            # episode_reward = get_utility_with_discount(offer=last_offer, prefer=rl_agent.prefer,
                            #                              condition=rl_agent.condition,
                            #                              domain_type=rl_agent.domain_type, issue_value=rl_agent.issue_value,
                            #                             time=rl_agent.relative_t,discount=rl_agent.discount)
                            avg_oppo += get_utility(offer=last_offer, prefer=opponent.prefer, condition=opponent.condition,
                                                    domain_type=opponent.domain_type, issue_value=opponent.issue_value)
                            #avg_oppo += get_utility_with_discount(offer=last_offer, prefer=opponent.prefer, condition=opponent.condition,
                            #                        domain_type=opponent.domain_type, issue_value=opponent.issue_value,time=opponent.relative_t,discount=opponent.discount)
                            succ_avg_reward += episode_reward
                            succ_avg_oppo += get_utility(offer=last_offer, prefer=opponent.prefer, condition=opponent.condition,
                                                         domain_type=opponent.domain_type, issue_value=opponent.issue_value)
                            # succ_avg_oppo += get_utility_with_discount(offer=last_offer, prefer=opponent.prefer,
                            #                              condition=opponent.condition,
                            #                              domain_type=opponent.domain_type, issue_value=opponent.issue_value,
                            #                             time=opponent.relative_t,discount=opponent.discount)
                            succ_counts += 1
                            if negotiation.domain_file not in succ_domains:
                                succ_domains.append(negotiation.domain_file)
                        break

                    if render:
                        print()

                    if current_player==1:
                        end=1
            if accept:
                print("Negotiation success")
            else:
                print("Negotiation failed")
            print("rl received reward: %f\n" % episode_reward)
            if accept == False:
                episode_round = negotiation.max_round #+ 1

            avg_reward += episode_reward
            avg_round += episode_round
        avg_reward /= eval_episodes
        avg_round /= eval_episodes
        avg_oppo /= eval_episodes
        if succ_counts != 0:
            succ_avg_reward /= succ_counts
            succ_avg_oppo /= succ_counts
        print("---------------------------------------")
        print("Evaluation over %d episodes: %f" % (eval_episodes, avg_reward))
        print("Average finished rounds: %f" % (avg_round))
        print("Opponent get average utility %f" % (avg_oppo))
        print("Number of successful negotiations:", succ_counts)
        print("The average utility of successful negotiations:", succ_avg_reward)
        print("Average utility of successfully negotiated opponents:", succ_avg_oppo)
        print("successful domains:", succ_domains)
        print("---------------------------------------")
        opponent_name=""
        if opponent.__class__.__name__ == "CUHKAgent" :
            opponent_name = "CUHKAgent"
        elif opponent.__class__.__name__ == "HardHeadedAgent" :
            opponent_name = "HardHeadedAgent"
        elif opponent.__class__.__name__ == "YXAgent" :
            opponent_name = "YXAgent"
        elif opponent.__class__.__name__ == "OMAC" :
            opponent_name = "OMAC"
        elif opponent.__class__.__name__ == "AgentLG" :
            opponent_name = "AgentLG"
        elif opponent.__class__.__name__ == "ParsAgent" :
            opponent_name = "ParsAgent"
        elif opponent.__class__.__name__ == "Caduceus" :
            opponent_name = "Caduceus"
        elif opponent.__class__.__name__ == "Atlas3" :
            opponent_name = "Atlas3"
        elif opponent.__class__.__name__ == "PonPokoAgent" :
            opponent_name = "PonPokoAgent"
        elif opponent.__class__.__name__ == "ParsCat" :
            opponent_name = "ParsCat"
        elif opponent.__class__.__name__ ==  "AgreeableAgent2018":
            opponent_name="AgreeableAgent2018"
        elif opponent.__class__.__name__ ==  "Agent36":
            opponent_name = "Agent36"

        return avg_reward,avg_round,avg_oppo,succ_counts,succ_avg_reward,succ_avg_oppo,opponent_name

    done = False
    for i in range(1, negotiation.max_round + 1):
        end = 0
        while end != 1:
            
            current_player = 1 - current_player
            if i == 1:
                negotiation.agents_pool[current_player].receive(last_offer)
            episode_timesteps += 1
            #current_player = 1 - i % 2
            negotiation.agents_pool[current_player].set_t(i)#time/max_round
            negotiation.agents_pool[1 - current_player].set_t(i)
            
            if negotiation.agents_pool[current_player].__class__.__name__ == "RLAgent":

                obs = negotiation.agents_pool[current_player].obs
                if render:
                    print("  RL agent's obs: ", rl_agent.obs)
                # if total_timesteps < start_timesteps:
                #     action = np.random.uniform(-1, 1, [1])
                # else:
                    #action = rl_agent.act()
                action =rl_agent.act()
                # last_utility 
                last_utility = 0.5 * (action + 1) * (rl_agent.u_max - rl_agent.u_min) + rl_agent.u_min
                #last_utility=last_utility*math.pow(rl_agent.discount,rl_agent.relative_t)
                rl_agent.s = last_utility
                last_offer = rl_agent.gen_offer()
            else:
                last_offer = negotiation.agents_pool[current_player].act()

            reward = 0.
            negotiation.agents_pool[1 - current_player].receive(last_offer)#Opponent received information


            if negotiation.agents_pool[current_player].__class__.__name__ != "RLAgent":
                new_obs = negotiation.agents_pool[1 - current_player].obs#Generate a new state where the opponent needs to change their state upon receiving new messages

            if episode_timesteps >= negotiation.max_round:
                done = True
                reward = -1
                new_obs = obs

            if negotiation.agents_pool[1 - current_player].accept:
                #print("The other agent accepts the end of the conversation！！！！！！！！！！！！")
                #print("agent name:", negotiation.agents_pool[1 - current_player].__class__.__name__)
                done = True
                #reward = get_utility(last_offer, rl_agent.prefer, rl_agent.condition, rl_agent.domain_type,rl_agent.issue_value)
                reward = get_utility_with_discount(last_offer, rl_agent.prefer, rl_agent.condition, rl_agent.domain_type,
                                     rl_agent.issue_value,rl_agent.relative_t, rl_agent.discount)

                new_obs = obs

            if (last_offer is None) and (
                negotiation.agents_pool[current_player].__class__.__name__ == "CUHKAgent" 
                or negotiation.agents_pool[current_player].__class__.__name__ == "HardHeadedAgent" 
                or negotiation.agents_pool[current_player].__class__.__name__ == "YXAgent" 
                or negotiation.agents_pool[current_player].__class__.__name__ == "OMAC" 
                or negotiation.agents_pool[current_player].__class__.__name__ == "AgentLG" 
                or negotiation.agents_pool[current_player].__class__.__name__ == "ParsAgent" 
                or negotiation.agents_pool[current_player].__class__.__name__ == "Caduceus" 
                or negotiation.agents_pool[current_player].__class__.__name__ == "Atlas3" 
                or negotiation.agents_pool[current_player].__class__.__name__ == "PonPokoAgent" 
                or negotiation.agents_pool[current_player].__class__.__name__ == "ParsCat" 
                or negotiation.agents_pool[current_player].__class__.__name__ == "AgreeableAgent2018"
                or negotiation.agents_pool[current_player].__class__.__name__ == "Agent36"):
                if negotiation.agents_pool[current_player].accept == True:
                    #print("The current agent accepts the end of the conversation！！！！！！！！！！！！")
                    #print("agent name:",negotiation.agents_pool[current_player].__class__.__name__ )
                    episode_round = i


                    done = True
                    #reward = negotiation.agents_pool[1 - current_player].utility_proposed[-1]
                    reward = negotiation.agents_pool[1 - current_player].utility_proposed[-1]

                elif negotiation.agents_pool[current_player].terminate == True:

                    done = True
                    reward = -1
            elif last_offer is None:
                print("Training code error existing: agent's offer is None.")
                exit(-1)

            done_bool = float(done)

            if obs is not None and new_obs is not None and (obs != new_obs or done):
                if path_length < max_path_length1:
                    observations.append(obs)
                    rewards.append(reward)
                    terminals.append(done_bool)
                    actions.append(action)
                    next_observations.append(new_obs)
                    
                    path_length += 1
                 

            episode_reward += reward

            total_timesteps += 1
            timesteps_since_eval += 1

            if done:
                break
            if current_player == 1:
                end = 1

    
    actions = np.array(actions)
    if len(actions.shape) == 1:
        actions = np.expand_dims(actions, 1)
    observations = np.array(observations)
    next_observations = np.array(next_observations)
    
    rewards = np.array(rewards)
    if len(rewards.shape) == 1:
        rewards = rewards.reshape(-1, 1)
    return dict(
        observations=observations,
        actions=actions,
        rewards=rewards,
        next_observations=next_observations,
        terminals=np.array(terminals).reshape(-1, 1),
    )


def deprecated_rollout(
        env,
        agent,
        max_path_length=np.inf,
        render=False,
        render_kwargs=None,
):
    """
    The following value for the following keys will be a 2D array, with the
    first dimension corresponding to the time dimension.
     - observations
     - actions
     - rewards
     - next_observations
     - terminals

    The next two elements will be lists of dictionaries, with the index into
    the list being the index into the time
     - agent_infos
     - env_infos
    """
    if render_kwargs is None:
        render_kwargs = {}
    observations = []
    actions = []
    rewards = []
    terminals = []
    agent_infos = []
    env_infos = []
    o = env.reset()
    agent.reset()
    next_o = None
    path_length = 0
    if render:
        env.render(**render_kwargs)
    while path_length < max_path_length:
        a, agent_info = agent.get_action(o)
        next_o, r, d, env_info = env.step(a)
        observations.append(o)
        rewards.append(r)
        terminals.append(d)
        actions.append(a)
        agent_infos.append(agent_info)
        env_infos.append(env_info)
        path_length += 1
        if d:
            break
        o = next_o
        if render:
            env.render(**render_kwargs)

    actions = np.array(actions)
    if len(actions.shape) == 1:
        actions = np.expand_dims(actions, 1)
    observations = np.array(observations)
    if len(observations.shape) == 1:
        observations = np.expand_dims(observations, 1)
        next_o = np.array([next_o])
    next_observations = np.vstack(
        (
            observations[1:, :],
            np.expand_dims(next_o, 0)
        )
    )
    return dict(
        observations=observations,
        actions=actions,
        rewards=np.array(rewards).reshape(-1, 1),
        next_observations=next_observations,
        terminals=np.array(terminals).reshape(-1, 1),
        agent_infos=agent_infos,
        env_infos=env_infos,
    )



def function_rollout(
        env,
        agent_fn,
        max_path_length=np.inf,
        render=False,
        render_kwargs=None,
):
    """
    The following value for the following keys will be a 2D array, with the
    first dimension corresponding to the time dimension.
     - observations
     - actions
     - rewards
     - next_observations
     - terminals

    The next two elements will be lists of dictionaries, with the index into
    the list being the index into the time
     - agent_infos
     - env_infos
    """
    if render_kwargs is None:
        render_kwargs = {}
    observations = []
    actions = []
    rewards = []
    terminals = []
    env_infos = []
    o = env.reset()
    next_o = None
    path_length = 0
    if render:
        env.render(**render_kwargs)
    while path_length < max_path_length:
        a = agent_fn(o)
        next_o, r, d, env_info = env.step(a)
        observations.append(o)
        rewards.append(r)
        terminals.append(d)
        actions.append(a)
        env_infos.append(env_info)
        path_length += 1
        if d:
            break
        o = next_o
        if render:
            env.render(**render_kwargs)

    actions = np.array(actions)
    if len(actions.shape) == 1:
        actions = np.expand_dims(actions, 1)
    observations = np.array(observations)
    if len(observations.shape) == 1:
        observations = np.expand_dims(observations, 1)
        next_o = np.array([next_o])
    next_observations = np.vstack(
        (
            observations[1:, :],
            np.expand_dims(next_o, 0)
        )
    )
    return dict(
        observations=observations,
        actions=actions,
        rewards=np.array(rewards).reshape(-1, 1),
        next_observations=next_observations,
        terminals=np.array(terminals).reshape(-1, 1),
        env_infos=env_infos,
    )

