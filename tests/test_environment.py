import sys, os; sys.path.insert(0, os.path.abspath("src"))
from environment import StackelbergEnv


def test_step_returns_reward_and_demand():
    env = StackelbergEnv()
    reward, demand = env.step(1)
    assert isinstance(reward, float)
    assert isinstance(demand, int)
    assert demand in env.demand_levels
