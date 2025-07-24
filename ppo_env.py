import gym
from gym import spaces
import numpy as np
from env import AIGCStackelbergEnv

class StackelbergGymEnv(gym.Env):
    """
    Gym wrapper cho môi trường Stackelberg: agent là ASP, action là price, state là lịch sử (price, participants).
    """
    def __init__(self, num_vmu=10, history_len=10, cost_per_step=1, p_min=1, p_max=20):
        super().__init__()
        self.env = AIGCStackelbergEnv(
            num_vmu=num_vmu,
            cost_per_step=cost_per_step,
            p_min=p_min,
            p_max=p_max
        )
        self.history_len = history_len
        # Observation: history_len x 2 (price, num_participants)
        self.observation_space = spaces.Box(
            low=0, high=10000, shape=(history_len * 2,), dtype=np.float32
        )
        # Action: chỉ 1 giá trị (price)
        self.action_space = spaces.Box(
            low=p_min, high=p_max, shape=(1,), dtype=np.float32
        )
        self.max_ep_len = 600
        self.reset()

    def reset(self):
        self.env.reset()
        # Bắt đầu với lịch sử toàn 0
        self.price_hist = [self.env.p_min] * self.history_len
        self.participant_hist = [0] * self.history_len
        self.t = 0
        return np.array(self.price_hist + self.participant_hist, dtype=np.float32)

    def step(self, action):
        # Agent chọn price
        price = float(np.clip(action[0], self.env.p_min, self.env.p_max))
        state, asp_util, _, info = self.env.step(price)
        num_participants = state[1]
        # Lưu lịch sử cho quan sát
        self.price_hist = self.price_hist[1:] + [price]
        self.participant_hist = self.participant_hist[1:] + [num_participants]
        self.t += 1
        done = self.t >= self.max_ep_len
        obs = np.array(self.price_hist + self.participant_hist, dtype=np.float32)
        return obs, asp_util, done, info
