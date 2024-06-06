import gym
import collections

import gym.spaces
import cv2
import numpy as np

class RepeatAction(gym.Wrapper):

    def __init__(self, env=None, repeat=4, fire_first=False):
        super(RepeatAction, self).__init__(env)
        self.repeat = repeat
        self.shape = env.observation_space.low.shape
        self.fire_first = fire_first
    
    def step(self, action):
        t_reward = 0.0
        done = False
        for i in range(self.repeat):
            obs, reward, done, info = self.env.step(action)
            t_reward += reward
            if done:
                break
        return obs, t_reward, done, info
    
    def reset(self):
        obs, _ = self.env.reset()  # Unpack only the observation from reset
        if self.fire_first:
            assert self.env.unwrapped.get_action_meanings()[1] == 'FIRE'
            obs, _, done, _ = self.env.step(1)  # Properly unpack the results from step
            if done:
                obs = self.env.reset()  # If done is True after FIRE, reset the environment again
        return obs



class PreprocessFrame(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, shape):
        super(PreprocessFrame, self).__init__(env)
        self.shape = (shape[2], shape[0], shape[1])
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=self.shape, dtype=np.float32)
    
    def observation(self, observation):
        new_frame = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
        resized_screen = cv2.resize(new_frame, self.shape[1:], interpolation=cv2.INTER_AREA)
        new_obs = np.array(resized_screen, dtype=np.uint8).reshape(self.shape)
        new_obs = new_obs / 255.0

        return new_obs
    

class StackFrames(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, repeat):
        super(StackFrames, self).__init__(env)
        self.observation_space = gym.spaces.Box(env.observation_space.low.repeat(repeat, axis=0),
                                                env.observation_space.high.repeat(repeat, axis=0),
                                                dtype=np.float32)
        self.stack = collections.deque(maxlen=repeat)
    
    def reset(self):
        self.stack.clear()
        obs, _ = self.env.reset()
        for _ in range(self.stack.maxlen):
            self.stack.append(obs)

        return np.array(self.stack).reshape(self.observation_space.low.shape)
    
    def observation(self, obs):
        self.stack.append(obs)

        return np.array(self.stack).reshape(self.observation_space.low.shape)
    

def make_env(env_name, shape=(42, 42, 1), repeat=4):
    env = gym.make(env_name)
    # env = RepeatAction(env, repeat)
    env = PreprocessFrame(env, shape)
    env = StackFrames(env, repeat)
    return env