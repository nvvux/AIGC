from env import AIGCStackelbergEnv
from model import ASPStackelbergAnalytical

def run_stackelberg_analytical():
    env = AIGCStackelbergEnv()
    asp_policy = ASPStackelbergAnalytical(env)
    best_price, best_steps, best_asp_util = asp_policy.search()
    print(f"[Stackelberg equilibrium]")
    print(f"Optimal price: {best_price:.2f}")
    print(f"ASP utility: {best_asp_util:.2f}")
    print(f"Sum VMU steps: {best_steps.sum():.1f}")
    print(f"Steps per VMU: {best_steps.round(1)}")
    return best_price, best_steps, best_asp_util
