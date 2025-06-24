import numpy as np
from config import *
from utils import satisfaction_func, gen_deltas, clamp

class AIGCStackelbergEnv:
    def __init__(self, num_vmu=NUM_VMU, cost_per_step=COST_PER_STEP, p_max=P_MAX):
        self.num_vmu = num_vmu
        self.COST_PER_STEP = cost_per_step
        self.P_MAX = p_max
        self.deltas = gen_deltas(num_vmu)
        self.S_MIN = S_MIN
        self.S_MAX = S_MAX
        self.RESOURCE_LIMIT = RESOURCE_LIMIT
        self.reset()
    def reset(self):
        self.deltas = gen_deltas(self.num_vmu)
        self.price = None
        self.steps = np.zeros(self.num_vmu)
        self.asp_utility = 0
        self.vmu_utils = np.zeros(self.num_vmu)
    def vmu_best_response(self, price):
        S_stars = []
        for k in range(self.num_vmu):
            S_star = (price - BETA) / (2 * ALPHA)
            S_star = clamp(S_star, self.S_MIN, self.S_MAX)
            S_stars.append(S_star)
        return np.array(S_stars)
    def compute_utilities(self, price, steps):
        vmu_utils = [satisfaction_func(S, self.deltas[k]) - price * S for k, S in enumerate(steps)]
        asp_util = np.sum((price - self.COST_PER_STEP) * steps)
        return np.array(vmu_utils), asp_util
    def step(self, price):
        steps = self.vmu_best_response(price)
        if np.sum(steps) > self.RESOURCE_LIMIT:
            steps = steps * self.RESOURCE_LIMIT / np.sum(steps)
        vmu_utils, asp_util = self.compute_utilities(price, steps)
        self.price = price
        self.steps = steps
        self.asp_utility = asp_util
        self.vmu_utils = vmu_utils
        return steps, vmu_utils, asp_util
