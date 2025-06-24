import gym
from gym import spaces
import numpy as np
from env import AIGCStackelbergEnv

class StackelbergGymEnv(gym.Env):
    def __init__(self, num_vmu=10, history_len=10, cost_per_step=1, p_max=20):
        super().__init__()
        self.env = AIGCStackelbergEnv(num_vmu=num_vmu, cost_per_step=cost_per_step, p_max=p_max)
        self.history_len = history_len
        self.observation_space = spaces.Box(low=0, high=10000, shape=(history_len*2,), dtype=np.float32)
        self.action_space = spaces.Box(low=self.env.COST_PER_STEP, high=self.env.P_MAX, shape=(1,), dtype=np.float32)
        self.max_ep_len = 600
        self.reset()
    def reset(self):
        self.env.reset()
        self.price_hist = [self.env.COST_PER_STEP] * self.history_len
        self.step_hist = [0] * self.history_len
        self.t = 0
        return np.array(self.price_hist + self.step_hist, dtype=np.float32)
    def step(self, action):
        price = float(np.clip(action[0], self.env.COST_PER_STEP, self.env.P_MAX))
        steps, _, asp_util = self.env.step(price)
        step_sum = steps.sum()
        self.price_hist = self.price_hist[1:] + [price]
        self.step_hist = self.step_hist[1:] + [step_sum]
        self.t += 1
        done = self.t >= self.max_ep_len
        return np.array(self.price_hist + self.step_hist, dtype=np.float32), asp_util, done, {}
