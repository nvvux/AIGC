import numpy as np

def run_random_baseline(env, episodes=600):
    asp_utils = []
    for i in range(episodes):
        p = np.random.uniform(env.COST_PER_STEP, env.P_MAX)
        steps, _, asp_util = env.step(p)
        asp_utils.append(asp_util)
    return np.array(asp_utils)

def run_greedy_baseline(env, episodes=600, p_init=None, delta=0.2):
    if p_init is None:
        p_curr = (env.COST_PER_STEP + env.P_MAX) / 2
    else:
        p_curr = p_init
    asp_utils = []
    last_util = None
    direction = 1  # 1: tăng, -1: giảm
    for i in range(episodes):
        # Chỉ thử tăng hoặc giảm (không thử tất cả cùng lúc)
        p_new = p_curr + direction * delta
        if env.COST_PER_STEP <= p_new <= env.P_MAX:
            steps, _, asp_util = env.step(p_new)
            if last_util is None or asp_util > last_util:
                p_curr = p_new
                last_util = asp_util
            else:
                direction *= -1  # Đổi hướng nếu giảm utility
        asp_utils.append(last_util if last_util is not None else 0)
    return np.array(asp_utils)
