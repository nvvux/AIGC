import numpy as np


class AIGCStackelbergEnv:
    """
    Mô phỏng Stackelberg game đúng paper: ASP chọn giá p, các VMU analytic S_k*.
    """

    def __init__(self,
                 num_vmu=10,
                 alpha=-0.000174,
                 beta=0.075,
                 gamma=4891.89,
                 cost_per_step=1.0,
                 p_min=1.0,
                 p_max=20.0,
                 s_min=120,
                 s_max=160):
        self.num_vmu = num_vmu
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.cost_per_step = cost_per_step
        self.p_min = p_min
        self.p_max = p_max
        self.s_min = s_min
        self.s_max = s_max
        self.reset()

    def reset(self):
        self.delta_k = np.random.uniform(100, 300, size=self.num_vmu)
        self.history = []
        return self._get_state()

    def _ssim(self, s):
        return self.alpha * s ** 2 + self.beta * s + self.gamma

    def _best_sk(self, delta_k, p):
        # Eq. (11) analytic solution, clipped in [s_min, s_max]
        if self.alpha == 0:
            return self.s_max
        s_star = (p / (2 * delta_k * self.alpha)) - (self.beta / (2 * self.alpha))
        return np.clip(s_star, self.s_min, self.s_max)

    def step(self, price):
        # price: float
        p = float(np.clip(price, self.p_min, self.p_max))
        s_stars = np.array([self._best_sk(dk, p) for dk in self.delta_k])
        ssim_vals = self._ssim(s_stars)
        utilities_k = self.delta_k * ssim_vals - p * s_stars
        participation = utilities_k >= 0
        num_participants = np.sum(participation)
        asp_reward = np.sum((p - self.cost_per_step) * s_stars[participation])
        # State: (price, num_participants)
        self.history.append((p, num_participants))
        return np.array([p, num_participants]), asp_reward, False, {}

    def _get_state(self):
        # Trả về lịch sử 10 lần gần nhất (p, num_participants)
        if len(self.history) < 10:
            return np.zeros((10, 2))
        return np.array(self.history[-10:])
