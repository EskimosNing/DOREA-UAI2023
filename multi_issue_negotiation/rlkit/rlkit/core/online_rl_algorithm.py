import abc

import gtimer as gt

from rlkit.core.rl_algorithm import BaseRLAlgorithm
from rlkit.data_management.replay_buffer import ReplayBuffer
from rlkit.samplers.data_collector import (
    PathCollector,
    StepCollector,
)
from negotiation import Negotiation

class OnlineRLAlgorithm(BaseRLAlgorithm, metaclass=abc.ABCMeta):
    def __init__(
            self,
            trainer,
            exploration_data_collector,
            evaluation_data_collector,
            replay_buffer: ReplayBuffer,
            batch_size,
            max_path_length,
            num_epochs,
            num_eval_steps_per_epoch,
            num_expl_steps_per_train_loop,
            num_trains_per_train_loop,
            num_train_loops_per_epoch=1,
            min_num_steps_before_training=0,
    ):
        super().__init__(
            trainer,
            exploration_data_collector,
            evaluation_data_collector,
            replay_buffer,
        )
        self.batch_size = batch_size
        self.max_path_length = max_path_length
        self.num_epochs = num_epochs
        self.num_eval_steps_per_epoch = num_eval_steps_per_epoch
        self.num_trains_per_train_loop = num_trains_per_train_loop
        self.num_train_loops_per_epoch = num_train_loops_per_epoch
        self.num_expl_steps_per_train_loop = num_expl_steps_per_train_loop
        self.min_num_steps_before_training = min_num_steps_before_training

        assert self.num_trains_per_train_loop >= self.num_expl_steps_per_train_loop, \
            'Online training presumes num_trains_per_train_loop >= num_expl_steps_per_train_loop'

    def _train(self):
        number_episodes=0
        self.training_mode(False)
        if self.min_num_steps_before_training > 0:
        
            self.expl_data_collector.collect_new_steps(
                self.max_path_length,
                self.min_num_steps_before_training,
                discard_incomplete_paths=False,
            )

            init_expl_paths = self.expl_data_collector.get_epoch_paths()
            self.replay_buffer.add_paths(init_expl_paths)
            self.expl_data_collector.end_epoch(-1)
            
            number_episodes = self.expl_data_collector.get_num_paths_total()
            numberOfTransition = self.expl_data_collector.get_num_steps_total()
            
            print('initial exploration')
            gt.stamp('initial exploration', unique=True)

        num_trains_per_expl_step = self.num_trains_per_train_loop // self.num_expl_steps_per_train_loop

        for epoch in gt.timed_for(
                range(self._start_epoch, self.num_epochs),
                save_itrs=True,
        ):
            # Collect data
            avg_reward, avg_round, avg_oppo, succ_counts, succ_avg_reward, succ_avg_oppo, opponent_name = self.eval_data_collector.collect_new_paths(

                self.max_path_length,
                self.num_eval_steps_per_epoch,
                discard_incomplete_paths=True,
            )

            number_episodes = self.expl_data_collector.get_num_paths_total()
            numberOfTransition = self.expl_data_collector.get_num_steps_total()
            print("evaluation sampling")
            
            gt.stamp('evaluation sampling')


            for _ in range(self.num_train_loops_per_epoch):
                for _ in range(self.num_expl_steps_per_train_loop):
                    
                    self.expl_data_collector.collect_new_steps(
                        self.max_path_length,
                        1,  
                        #render=False,
                        discard_incomplete_paths=False,
                    )
                    gt.stamp('exploration sampling', unique=False)

                    self.training_mode(True)
                    for _ in range(num_trains_per_expl_step):
                        train_data = self.replay_buffer.random_batch(
                            self.batch_size)
                        stats=self.trainer.train(train_data)
                        number_episodes = self.expl_data_collector.get_num_paths_total()
                        numberOfTransition = self.expl_data_collector.get_num_steps_total()
                        
                    gt.stamp('training', unique=False)
                    self.training_mode(False)

            new_expl_paths = self.expl_data_collector.get_epoch_paths()
            self.replay_buffer.add_paths(new_expl_paths)
            gt.stamp('data storing', unique=False)
            number_episodes = self.expl_data_collector.get_num_paths_total()
            numberOfTransition = self.expl_data_collector.get_num_steps_total()
            


            self._end_epoch(epoch)
