from collections import deque, OrderedDict
from functools import partial

import numpy as np
import random
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.samplers.data_collector.base import PathCollector
from rlkit.samplers.rollout_functions import rollout, function_rollout
from rlkit.data_management.path_builder import PathBuilder
from negotiation import Negotiation
from utils import get_utility

class MdpPathCollector(PathCollector):
    def __init__(
            self,
            max_round,
            domain_type,
            domain,
            rl_agent,
            opponent,
            opponents_pool,#
            start_timesteps,
            policy,
            episode_timesteps=1,
            max_num_epoch_paths_saved=None,
            render=False,
            render_kwargs=None,
            rollout_fn=rollout,
            save_env_in_snapshot=True,
    ):
        if render_kwargs is None:
            render_kwargs = {}
        self.max_round=max_round
        self.domain_type=domain_type
        self.domain=domain
        self.rl_agent=rl_agent
        self.opponents_pool=opponents_pool
        #opp_len=len(self.opponents_pool)

        #select_num=random.randint(0,opp_len-1)
        self.opponent=opponent#opponents_pool
        self.start_timesteps=start_timesteps
        self._policy = policy
        self._max_num_epoch_paths_saved = max_num_epoch_paths_saved
        self._epoch_paths = deque(maxlen=self._max_num_epoch_paths_saved)
        self._render = render
        self._render_kwargs = render_kwargs
        self._rollout_fn = rollout_fn
        self.episode_timesteps=episode_timesteps
        self._num_steps_total = 0
        self._num_paths_total = 0

        self._save_env_in_snapshot = save_env_in_snapshot
    def get_num_steps_total(self):
        return self._num_steps_total
    def get_num_paths_total(self):
        return self._num_paths_total
    def _start_new_rollout(self):
        self._current_path_builder = PathBuilder()
        self.episode_timesteps=1
        #self._obs = self._env.reset()
    def collect_new_steps(
            self,
            max_path_length,
            num_steps,
            discard_incomplete_paths,
    ):


        for j in range(num_steps):
            print(j)
            self.collect_one_step(max_path_length, discard_incomplete_paths)#,negotiation,self.rl_agent,self.opponent
    def chage_opponent(self):
        opp_len = len(self.opponents_pool)
        select_num = random.randint(0, opp_len - 1)
        self.opponent = self.opponents_pool[select_num]
    def set_opponent(self,opponent_name):
        self.opponent=opponent_name
    def get_opponent(self):
        return self.opponent
    def collect_one_step(
            self,
            max_path_length,
            discard_incomplete_paths,
            render=False
    ):
        raw_obs = []
        raw_next_obs = []
        observations = []
        actions = []
        rewards = []
        terminals = []
        agent_infos = []
        env_infos = []
        next_observations = []
        path_length = 0
        self._policy.reset()  # policy
        #######

        total_timesteps = 0
        timesteps_since_eval = 0
        episode_num = 0
        episode_num_last_eval = 0
        reward_records = []


        if self.rl_agent.t==0 and self.opponent.t==0:
            self.negotiation = Negotiation(
                max_round=self.max_round,
                domain_type=self.domain_type,
                domain_file=self.domain,
                train_mode=True
            )
            self.negotiation.add(self.rl_agent)
            self.negotiation.add(self.opponent)
            self.negotiation.reset()
            self._start_new_rollout()

        #negotiation.reset()

        episode_reward = 0
        #episode_timesteps = 0
        episode_num += 1
        episode_num_last_eval += 1

        last_offer = None
        current_player = 0
        obs = None
        action = None
        new_obs = None
        done = False

        while self.episode_timesteps < self.negotiation.max_round+1:
            i=self.episode_timesteps
       

            if i == 1:  
                self.negotiation.agents_pool[current_player].receive(last_offer)
            self.episode_timesteps += 1  
            current_player = 1 - i % 2  
            self.negotiation.agents_pool[current_player].set_t(i)  # time/max_round
            self.negotiation.agents_pool[1 - current_player].set_t(i)
            """if RLAgent"""
            if self.negotiation.agents_pool[current_player].__class__.__name__ == "RLAgent":

                obs = self.negotiation.agents_pool[current_player].obs
                if render:
                    print("  RL agent's obs: ", self.rl_agent.obs)
                # if total_timesteps < start_timesteps:
                #     action = np.random.uniform(-1, 1, [1])
                # else:
                    # action = rl_agent.act()
                action = self.rl_agent.act()
                
                last_utility = 0.5 * (action + 1) * (self.rl_agent.u_max - self.rl_agent.u_min) + self.rl_agent.u_min
                self.rl_agent.s = last_utility
                last_offer = self.rl_agent.gen_offer()
            else:
                last_offer = self.negotiation.agents_pool[current_player].act()

            reward = 0.
            self.negotiation.agents_pool[1 - current_player].receive(last_offer)  

            if self.negotiation.agents_pool[current_player].__class__.__name__ != "RLAgent":
                new_obs = self.negotiation.agents_pool[1 - current_player].obs  

            if self.episode_timesteps >= self.negotiation.max_round+1 :
                done = True
                reward = -1
                new_obs = obs

            if self.negotiation.agents_pool[1 - current_player].accept:
                # print("The other agent accepts the end of the conversation！！！！！！！！！！！！")
                # print("agent name:", negotiation.agents_pool[1 - current_player].__class__.__name__)
                done = True
                reward = get_utility(last_offer, self.rl_agent.prefer, self.rl_agent.condition, self.rl_agent.domain_type,
                                     self.rl_agent.issue_value)
                new_obs = obs

            if (last_offer is None) and (
                    self.negotiation.agents_pool[current_player].__class__.__name__ == "CUHKAgent" or
                    self.negotiation.agents_pool[
                        current_player].__class__.__name__ == "HardHeadedAgent" \
                    or self.negotiation.agents_pool[current_player].__class__.__name__ == "YXAgent" or
                    self.negotiation.agents_pool[
                        current_player].__class__.__name__ == "OMAC" \
                    or self.negotiation.agents_pool[current_player].__class__.__name__ == "AgentLG" or
                    self.negotiation.agents_pool[
                        current_player].__class__.__name__ == "ParsAgent" \
                    or self.negotiation.agents_pool[current_player].__class__.__name__ == "Caduceus" or
                    self.negotiation.agents_pool[
                        current_player].__class__.__name__ == "Atlas3" or self.negotiation.agents_pool[
                        current_player].__class__.__name__ == "PonPokoAgent" or self.negotiation.agents_pool[
                        current_player].__class__.__name__ == "ParsCat"):
                if self.negotiation.agents_pool[current_player].accept == True:
                    # print("The current agent accepts the end of the conversation！！！！！！！！！！！！")
                    # print("agent name:",negotiation.agents_pool[current_player].__class__.__name__ )
                    episode_round = i

                    done = True
                    reward = self.negotiation.agents_pool[1 - current_player].utility_proposed[-1]

                elif self.negotiation.agents_pool[current_player].terminate == True:

                    done = True
                    reward = -1
            elif last_offer is None:
                print("Training code error existing: agent's offer is None.")
                exit(-1)

            done_bool = float(done)

            if obs is not None and new_obs is not None and (obs != new_obs or done):
                #if path_length < max_path_length:
                self._current_path_builder.add_all(
                        observations=obs,
                        actions=action,
                        rewards=reward,
                        next_observations=new_obs,
                        terminals=done_bool,
                )

            episode_reward += reward

            total_timesteps += 1
            timesteps_since_eval += 1

            if done or len(self._current_path_builder)>=max_path_length or self.episode_timesteps>=self.negotiation.max_round+1:
                self._handle_rollout_ending(max_path_length,discard_incomplete_paths)
                self.negotiation.reset()
                self._start_new_rollout()
                break
    def _handle_rollout_ending(
            self,
            max_path_length,
            discard_incomplete_paths
    ):
        if len(self._current_path_builder) > 0:
            path = self._current_path_builder.get_all_stacked()
            path_len = len(path['actions'])
            self._epoch_paths.append(path)
            self._num_paths_total += 1
            self._num_steps_total += path_len
    def collect_new_paths(
            self,
            max_path_length,
            num_steps,
            discard_incomplete_paths,
    ):
        paths = []
        num_steps_collected = 0
        episodes=0
        while num_steps_collected < num_steps:
            max_path_length_this_loop = min(  # Do not go over num_steps
                max_path_length,
                num_steps - num_steps_collected,
            )
            #domain_type="DISCRETE"
            negotiation=Negotiation(
                max_round=self.max_round,
                domain_type=self.domain_type,
                domain_file=self.domain,
                train_mode=True
            )
            if self._render:#eval
                avg_reward,avg_round,avg_oppo,succ_counts,succ_avg_reward,succ_avg_oppo,opponent_name = self._rollout_fn(
                    negotiation,
                    self._policy,
                    self.rl_agent,
                    self.opponent,
                    self.start_timesteps,

                    max_path_length=max_path_length_this_loop,
                    #opponents_pool=self.opponents_pool,
                    render=self._render,
                    render_kwargs=self._render_kwargs,

                )
                return avg_reward,avg_round,avg_oppo,succ_counts,succ_avg_reward,succ_avg_oppo,opponent_name
            else:
                path = self._rollout_fn(
                    negotiation,
                    self._policy,
                    self.rl_agent,
                    self.opponent,
                    self.start_timesteps,

                    max_path_length=max_path_length_this_loop,
                    render=self._render,
                    render_kwargs=self._render_kwargs,

                )
                #episodes=+1
                path_len = len(path['actions'])

                # if (
                #         path_len != max_path_length
                #         and not path['terminals'][-1]
                #         and discard_incomplete_paths
                # ):
                #     break
                num_steps_collected += path_len
                paths.append(path)
        self._num_paths_total += len(paths)
        self._num_steps_total += num_steps_collected
        self._epoch_paths.extend(paths)
        return paths

    def get_epoch_paths(self):
        return self._epoch_paths

    def end_epoch(self, epoch):
        self._epoch_paths = deque(maxlen=self._max_num_epoch_paths_saved)

    def get_diagnostics(self):
        path_lens = [len(path['actions']) for path in self._epoch_paths]
        stats = OrderedDict([
            ('num steps total', self._num_steps_total),
            ('num paths total', self._num_paths_total),
        ])
        stats.update(create_stats_ordered_dict(
            "path length",
            path_lens,
            always_show_all_stats=True,
        ))
        return stats

    def get_snapshot(self):
        snapshot_dict = dict(
            policy=self._policy,
        )

        return snapshot_dict


class CustomMDPPathCollector(PathCollector):
    def __init__(
        self,
        max_num_epoch_paths_saved=None,
        render=False,
        render_kwargs=None,
    ):
        if render_kwargs is None:
            render_kwargs = {}
        self._max_num_epoch_paths_saved = max_num_epoch_paths_saved
        self._epoch_paths = deque(maxlen=self._max_num_epoch_paths_saved)
        self._render = render
        self._render_kwargs = render_kwargs

        self._num_steps_total = 0
        self._num_paths_total = 0

    def collect_new_paths(
            self, policy_fn, max_path_length, 
            num_steps, discard_incomplete_paths
        ):
        paths = []
        num_steps_collected = 0
        while num_steps_collected < num_steps:
            max_path_length_this_loop = min(  # Do not go over num_steps
                max_path_length,
                num_steps - num_steps_collected,
            )
            path = function_rollout(
                self._env,
                policy_fn,
                max_path_length=max_path_length_this_loop,
            )
            path_len = len(path['actions'])
            if (
                    path_len != max_path_length
                    and not path['terminals'][-1]
                    and discard_incomplete_paths
            ):
                break
            num_steps_collected += path_len
            paths.append(path)
        self._num_paths_total += len(paths)
        self._num_steps_total += num_steps_collected
        self._epoch_paths.extend(paths)
        return paths
    
    def get_epoch_paths(self):
        return self._epoch_paths

    def end_epoch(self, epoch):
        self._epoch_paths = deque(maxlen=self._max_num_epoch_paths_saved)

    def get_diagnostics(self):
        path_lens = [len(path['actions']) for path in self._epoch_paths]
        stats = OrderedDict([
            ('num steps total', self._num_steps_total),
            ('num paths total', self._num_paths_total),
        ])
        stats.update(create_stats_ordered_dict(
            "path length",
            path_lens,
            always_show_all_stats=True,
        ))
        return stats



class GoalConditionedPathCollector(MdpPathCollector):
    def __init__(
            self,
            *args,
            observation_key='observation',
            desired_goal_key='desired_goal',
            goal_sampling_mode=None,
            **kwargs
    ):
        def obs_processor(o):
            return np.hstack((o[observation_key], o[desired_goal_key]))

        rollout_fn = partial(
            rollout,
            preprocess_obs_for_policy_fn=obs_processor,
        )
        super().__init__(*args, rollout_fn=rollout_fn, **kwargs)
        self._observation_key = observation_key
        self._desired_goal_key = desired_goal_key
        self._goal_sampling_mode = goal_sampling_mode

    def collect_new_paths(self, *args, **kwargs):
        self._env.goal_sampling_mode = self._goal_sampling_mode
        return super().collect_new_paths(*args, **kwargs)

    def get_snapshot(self):
        snapshot = super().get_snapshot()
        snapshot.update(
            observation_key=self._observation_key,
            desired_goal_key=self._desired_goal_key,
        )
        return snapshot


class ObsDictPathCollector(MdpPathCollector):
    def __init__(
            self,
            *args,
            observation_key='observation',
            **kwargs
    ):
        def obs_processor(obs):
            return obs[observation_key]

        rollout_fn = partial(
            rollout,
            preprocess_obs_for_policy_fn=obs_processor,
        )
        super().__init__(*args, rollout_fn=rollout_fn, **kwargs)
        self._observation_key = observation_key

    def get_snapshot(self):
        snapshot = super().get_snapshot()
        snapshot.update(
            observation_key=self._observation_key,
        )
        return snapshot


class VAEWrappedEnvPathCollector(GoalConditionedPathCollector):
    def __init__(
            self,
            env,
            policy,
            decode_goals=False,
            **kwargs
    ):
        """Expects env is VAEWrappedEnv"""
        super().__init__(env, policy, **kwargs)
        self._decode_goals = decode_goals

    def collect_new_paths(self, *args, **kwargs):
        self._env.decode_goals = self._decode_goals
        return super().collect_new_paths(*args, **kwargs)
