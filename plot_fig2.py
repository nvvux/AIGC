import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
""""
import numpy as np
import matplotlib.pyplot as plt
from env import AIGCStackelbergEnv
from model import ASPStackelbergAnalytical
from baselines import run_random_baseline, run_greedy_baseline
from train_ppo import train_ppo_agent, test_ppo_agent
from train_tppo import train_tppo_agent, test_tppo_agent
from config import EPISODES

# ---- (a) Convergence analysis ----
env = AIGCStackelbergEnv()
stackelberg = ASPStackelbergAnalytical(env)
_, _, asp_opt = stackelberg.search()
stackelberg_util = np.ones(EPISODES) * asp_opt

ppo_model = train_ppo_agent()
ppo_utils = test_ppo_agent(ppo_model)
tppo_model = train_tppo_agent()
tppo_utils = test_tppo_agent(tppo_model)
random_utils = run_random_baseline(env)
greedy_utils = run_greedy_baseline(env)

plt.figure(figsize=(7,4))
plt.plot(stackelberg_util, 'k--', label="Stackelberg equilibrium")
plt.plot(tppo_utils, label="Transformer-based PPO")
plt.plot(ppo_utils, label="PPO")
plt.plot(greedy_utils, label="Greedy")
plt.plot(random_utils, label="Random")
plt.xlabel("Number of iterations")
plt.ylabel("Utility of ASP")
plt.legend()
plt.tight_layout()
plt.title("(a) Convergence analysis")
plt.show()

# ---- (b), (d) Utility và optimal price vs. cost ----
cost_list = [1, 2, 3, 4, 5]
asp_utils_stack = []
asp_utils_tppo = []
asp_utils_ppo = []
asp_utils_greedy = []
asp_utils_random = []
optimal_prices = []

for cost in cost_list:
    env = AIGCStackelbergEnv(cost_per_step=cost)
    stackelberg = ASPStackelbergAnalytical(env)
    p_star, _, asp_opt = stackelberg.search()
    asp_utils_stack.append(asp_opt)
    optimal_prices.append(p_star)
    tppo_model = train_tppo_agent(cost_per_step=cost)
    asp_utils_tppo.append(np.mean(test_tppo_agent(tppo_model, cost_per_step=cost)))
    ppo_model = train_ppo_agent(cost_per_step=cost)
    asp_utils_ppo.append(np.mean(test_ppo_agent(ppo_model, cost_per_step=cost)))
    asp_utils_greedy.append(run_greedy_baseline(env).mean())
    asp_utils_random.append(run_random_baseline(env).mean())

# (b) Utility vs cost
plt.figure(figsize=(7,4))
w = 0.15
x = np.array(cost_list)
plt.bar(x-w*2, asp_utils_stack, width=w, label='Stackelberg')
plt.bar(x-w, asp_utils_tppo, width=w, label='TPPO')
plt.bar(x, asp_utils_ppo, width=w, label='PPO')
plt.bar(x+w, asp_utils_greedy, width=w, label='Greedy')
plt.bar(x+2*w, asp_utils_random, width=w, label='Random')
plt.xlabel("Unit cost of diffusion steps")
plt.ylabel("Utility of ASP")
plt.legend()
plt.title("(b) Utility vs. cost")
plt.show()

plt.figure(figsize=(7,4))
plt.plot(cost_list, optimal_prices, marker='s', label='Optimal price (Stackelberg)')
plt.xlabel("Unit cost of diffusion steps")
plt.ylabel("Optimal price")
plt.legend()
plt.title("(b) Pricing strategy vs. cost")
plt.show()

# (d) Utility vs. cost (line)
plt.figure(figsize=(7,4))
plt.plot(cost_list, asp_utils_stack, 's-', label='Stackelberg')
plt.plot(cost_list, asp_utils_tppo, 'o-', label='TPPO')
plt.plot(cost_list, asp_utils_ppo, 'd-', label='PPO')
plt.plot(cost_list, asp_utils_greedy, '^-', label='Greedy')
plt.plot(cost_list, asp_utils_random, 'x-', label='Random')
plt.xlabel("Unit cost of diffusion steps")
plt.ylabel("Utility of ASP")
plt.legend()
plt.title("(d) Utility vs. cost")
plt.show()

# ---- (c) Utility vs. number of VMUs ----
num_vmu_list = [2, 4, 6, 8, 10]
asp_utils_stack = []
asp_utils_tppo = []
asp_utils_ppo = []
asp_utils_greedy = []
asp_utils_random = []

for n in num_vmu_list:
    env = AIGCStackelbergEnv(num_vmu=n)
    stackelberg = ASPStackelbergAnalytical(env)
    _, _, asp_opt = stackelberg.search()
    asp_utils_stack.append(asp_opt)
    tppo_model = train_tppo_agent(num_vmu=n)
    asp_utils_tppo.append(np.mean(test_tppo_agent(tppo_model, num_vmu=n)))
    ppo_model = train_ppo_agent(num_vmu=n)
    asp_utils_ppo.append(np.mean(test_ppo_agent(ppo_model, num_vmu=n)))
    asp_utils_greedy.append(run_greedy_baseline(env).mean())
    asp_utils_random.append(run_random_baseline(env).mean())

plt.figure(figsize=(7,4))
plt.plot(num_vmu_list, asp_utils_stack, 's-', label='Stackelberg')
plt.plot(num_vmu_list, asp_utils_tppo, 'o-', label='TPPO')
plt.plot(num_vmu_list, asp_utils_ppo, 'd-', label='PPO')
plt.plot(num_vmu_list, asp_utils_greedy, '^-', label='Greedy')
plt.plot(num_vmu_list, asp_utils_random, 'x-', label='Random')
plt.xlabel('Number of VMUs')
plt.ylabel('Utility of ASP')
plt.legend()
plt.title("(c) Utility vs. number of VMUs")
plt.show()
"""


import numpy as np
import matplotlib.pyplot as plt
import torch

from env import AIGCStackelbergEnv
from model import ASPStackelbergAnalytical
from baselines import run_random_baseline, run_greedy_baseline
from train_ppo import train_ppo_agent, test_ppo_agent
from train_tppo import train_tppo_agent, test_tppo_agent
from config import EPISODES

# === SETUP ===
SEED = 42
N_REPEAT = 5           # Số lần chạy mỗi baseline
ROLL_WINDOW = 20

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        import random
        random.seed(seed)
    except:
        pass

def rolling_mean(x, window=20):
    x = np.asarray(x)
    if x.ndim == 2:   # N_REPEAT x T
        x = x.mean(axis=0)
    return np.convolve(x, np.ones(window)/window, mode='valid')

# === (a) Convergence analysis ===
set_seed(SEED)
env = AIGCStackelbergEnv()
stackelberg = ASPStackelbergAnalytical(env)
_, _, asp_opt = stackelberg.search()
stackelberg_util = np.ones(EPISODES) * asp_opt

def baseline_repeat(func, *args, **kwargs):
    out = []
    for i in range(N_REPEAT):
        set_seed(SEED+i)
        out.append(func(*args, **kwargs))
    return np.array(out)   # N_REPEAT x T

ppo_utils    = baseline_repeat(lambda: test_ppo_agent(train_ppo_agent()))
tppo_utils   = baseline_repeat(lambda: test_tppo_agent(train_tppo_agent()))
greedy_utils = baseline_repeat(lambda: run_greedy_baseline(env))
random_utils = baseline_repeat(lambda: run_random_baseline(env))

plt.figure(figsize=(7,4))
plt.plot(stackelberg_util[ROLL_WINDOW-1:], 'k--', label="Stackelberg equilibrium")
plt.plot(rolling_mean(tppo_utils, ROLL_WINDOW), label="Transformer-based PPO")
plt.plot(rolling_mean(ppo_utils, ROLL_WINDOW), label="PPO")
plt.plot(rolling_mean(greedy_utils, ROLL_WINDOW), label="Greedy")
plt.plot(rolling_mean(random_utils, ROLL_WINDOW), label="Random")
plt.xlabel("Number of iterations")
plt.ylabel("Utility of ASP")
plt.legend()
plt.tight_layout()
plt.title("(a) Convergence analysis")
plt.show()

# === (b), (d) Utility và optimal price vs. cost ===
cost_list = [1, 2, 3, 4, 5]
asp_utils_stack = []
asp_utils_tppo = []
asp_utils_ppo = []
asp_utils_greedy = []
asp_utils_random = []
optimal_prices = []

for cost in cost_list:
    env = AIGCStackelbergEnv(cost_per_step=cost)
    stackelberg = ASPStackelbergAnalytical(env)
    p_star, _, asp_opt = stackelberg.search()
    asp_utils_stack.append(asp_opt)
    optimal_prices.append(p_star)

    tppo_tmp = []
    ppo_tmp = []
    greedy_tmp = []
    random_tmp = []
    for i in range(N_REPEAT):
        set_seed(SEED+i)
        tppo_model = train_tppo_agent(cost_per_step=cost)
        tppo_tmp.append(np.mean(test_tppo_agent(tppo_model, cost_per_step=cost)))
        ppo_model = train_ppo_agent(cost_per_step=cost)
        ppo_tmp.append(np.mean(test_ppo_agent(ppo_model, cost_per_step=cost)))
        greedy_tmp.append(run_greedy_baseline(env).mean())
        random_tmp.append(run_random_baseline(env).mean())
    asp_utils_tppo.append(np.mean(tppo_tmp))
    asp_utils_ppo.append(np.mean(ppo_tmp))
    asp_utils_greedy.append(np.mean(greedy_tmp))
    asp_utils_random.append(np.mean(random_tmp))

# (b) Utility vs cost
plt.figure(figsize=(7,4))
w = 0.15
x = np.array(cost_list)
plt.bar(x-w*2, asp_utils_stack, width=w, label='Stackelberg')
plt.bar(x-w, asp_utils_tppo, width=w, label='TPPO')
plt.bar(x, asp_utils_ppo, width=w, label='PPO')
plt.bar(x+w, asp_utils_greedy, width=w, label='Greedy')
plt.bar(x+2*w, asp_utils_random, width=w, label='Random')
plt.xlabel("Unit cost of diffusion steps")
plt.ylabel("Utility of ASP")
plt.legend()
plt.title("(b) Utility vs. cost")
plt.show()

plt.figure(figsize=(7,4))
plt.plot(cost_list, optimal_prices, marker='s', label='Optimal price (Stackelberg)')
plt.xlabel("Unit cost of diffusion steps")
plt.ylabel("Optimal price")
plt.legend()
plt.title("(b) Pricing strategy vs. cost")
plt.show()

# (d) Utility vs. cost (line)
plt.figure(figsize=(7,4))
plt.plot(cost_list, asp_utils_stack, 's-', label='Stackelberg')
plt.plot(cost_list, asp_utils_tppo, 'o-', label='TPPO')
plt.plot(cost_list, asp_utils_ppo, 'd-', label='PPO')
plt.plot(cost_list, asp_utils_greedy, '^-', label='Greedy')
plt.plot(cost_list, asp_utils_random, 'x-', label='Random')
plt.xlabel("Unit cost of diffusion steps")
plt.ylabel("Utility of ASP")
plt.legend()
plt.title("(d) Utility vs. cost")
plt.show()

# === (c) Utility vs. number of VMUs ===
num_vmu_list = [2, 4, 6, 8, 10]
asp_utils_stack = []
asp_utils_tppo = []
asp_utils_ppo = []
asp_utils_greedy = []
asp_utils_random = []

for n in num_vmu_list:
    env = AIGCStackelbergEnv(num_vmu=n)
    stackelberg = ASPStackelbergAnalytical(env)
    _, _, asp_opt = stackelberg.search()
    asp_utils_stack.append(asp_opt)

    tppo_tmp = []
    ppo_tmp = []
    greedy_tmp = []
    random_tmp = []
    for i in range(N_REPEAT):
        set_seed(SEED+i)
        tppo_model = train_tppo_agent(num_vmu=n)
        tppo_tmp.append(np.mean(test_tppo_agent(tppo_model, num_vmu=n)))
        ppo_model = train_ppo_agent(num_vmu=n)
        ppo_tmp.append(np.mean(test_ppo_agent(ppo_model, num_vmu=n)))
        greedy_tmp.append(run_greedy_baseline(env).mean())
        random_tmp.append(run_random_baseline(env).mean())
    asp_utils_tppo.append(np.mean(tppo_tmp))
    asp_utils_ppo.append(np.mean(ppo_tmp))
    asp_utils_greedy.append(np.mean(greedy_tmp))
    asp_utils_random.append(np.mean(random_tmp))

plt.figure(figsize=(7,4))
plt.plot(num_vmu_list, asp_utils_stack, 's-', label='Stackelberg')
plt.plot(num_vmu_list, asp_utils_tppo, 'o-', label='TPPO')
plt.plot(num_vmu_list, asp_utils_ppo, 'd-', label='PPO')
plt.plot(num_vmu_list, asp_utils_greedy, '^-', label='Greedy')
plt.plot(num_vmu_list, asp_utils_random, 'x-', label='Random')
plt.xlabel('Number of VMUs')
plt.ylabel('Utility of ASP')
plt.legend()
plt.title("(c) Utility vs. number of VMUs")
plt.show()
