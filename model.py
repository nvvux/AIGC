import numpy as np

class ASPStackelbergAnalytical:
    def __init__(self, env, n_grid=200):
        self.env = env
        self.p_grid = np.linspace(self.env.COST_PER_STEP, self.env.P_MAX, n_grid)
    def search(self):
        best_util = -np.inf
        best_p = None
        best_steps = None
        for p in self.p_grid:
            steps, _, asp_util = self.env.step(p)
            if asp_util > best_util:
                best_util = asp_util
                best_p = p
                best_steps = steps
        return best_p, best_steps, best_util
