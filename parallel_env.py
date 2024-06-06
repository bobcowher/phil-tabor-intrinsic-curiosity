import os
import gym
import torch.multiprocessing as mp
from model import ActorCritic
from shared_adam import SharedAdam
from worker import worker

os.environ['OMP_NUM_THREADS'] = '1'

class ParallelEnv:
    def __init__(self, env_id, global_idx, input_shape, n_actions, num_threads):
        names = [str(i) for i in range(num_threads)]

        global_actor_critic = ActorCritic(input_shape, n_actions)
        global_actor_critic.share_memory()
        global_optim = SharedAdam(global_actor_critic.parameters(), lr=1e-4)

        self.ps = [mp.Process(target=worker, args=(name, input_shape, n_actions, 
                                                   global_actor_critic, global_optim, 
                                                   env_id, num_threads, global_idx)) for name in names]
    
    def start_threads(self):
        [p.start() for p in self.ps]
        [p.join() for p in self.ps]


