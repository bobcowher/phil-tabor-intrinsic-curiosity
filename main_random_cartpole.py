import os
import torch.multiprocessing as mp
import time
import gymnasium as gym

os.environ['SET_NUM_THREADS'] = '1'

class ParallelEnv:
    def __init__(self, thread_count):

        self.threads = []
        
        for thread in range(thread_count):
            self.threads.append(mp.Process(target=worker, args=(f'{thread}',)))
    
    def start_threads(self):
        for thread in self.threads:
            thread.start()
        
        for thread in self.threads:
            thread.join()

def worker(name):
    for i in range(5):
        env = gym.make('CartPole-v1')

        done = False
        episode_reward = 0
        observation, info = env.reset()

        while not done:
            action = env.action_space.sample()
            observation, reward, done, _, _ = env.step(action)
            episode_reward += reward


        print(f"Completed episode {i} on thread {name} - Score: {episode_reward}")
        # for i in range(10):
    #     print(f"Thread {name} - iteration {i}")
    #     time.sleep(2)


if __name__ == '__main__':
    mp.set_start_method('spawn')
    thread_count = 4
    
    parallel_env = ParallelEnv(thread_count=thread_count)

    parallel_env.start_threads()


    # process = mp.Process(target=worker, args=('dale',))
    # process.start()
    # process.join()