import os
import gym
import torch.multiprocessing

os.environ['OMP_NUM_THREADS'] = '1'

class ParallelEnv:
    def __init__(self, env_id, num_threads):
        names = [str(i) for i in range(num_threads)]

        self.ps = [mp.Process(target=worker, args=(name, env_id)) for name in names]

        [p.start() for p in self.ps]
        [p.join() for p in self.ps]


def worker(name, env_id):
    env=gym.make(env_id)
    episode, max_eps, scores = 0, 10, []

    while episode < max_eps:
        obs = env.reset()
        score, done = 0, False
        wyh