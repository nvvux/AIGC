import numpy as np

def run_random_baseline(env, episodes=600):
    asp_utils = []
    for i in range(episodes):
        p = np.random.uniform(env.COST_PER_STEP, env.P_MAX)
        steps, _, asp_util = env.step(p)
        asp_utils.append(asp_util)
    return np.array(asp_utils)

def run_greedy_baseline(env, episodes=600, grid_size=100):
    asp_utils = []
    for ep in range(episodes):
        env.reset()  # random lại delta mỗi episode!
        best_util = -np.inf
        best_p = env.COST_PER_STEP
        for p in np.linspace(env.COST_PER_STEP, env.P_MAX, grid_size):
            steps, _, asp_util = env.step(p)
            if asp_util > best_util:
                best_util = asp_util
                best_p = p
        asp_utils.append(best_util)
    return np.array(asp_utils)

