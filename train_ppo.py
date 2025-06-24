# train_ppo.py

from ppo_env import StackelbergGymEnv
from stable_baselines3 import PPO

def train_ppo_agent(num_vmu=10, cost_per_step=1, p_max=20, history_len=10):
    env = StackelbergGymEnv(num_vmu=num_vmu, history_len=history_len, cost_per_step=cost_per_step, p_max=p_max)
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=200_000)
    model.save("ppo_stackelberg")
    return model

def test_ppo_agent(model, num_vmu=10, cost_per_step=1, p_max=20, history_len=10):
    env = StackelbergGymEnv(num_vmu=num_vmu, history_len=history_len, cost_per_step=cost_per_step, p_max=p_max)
    obs = env.reset()
    ppo_utils = []
    for i in range(600):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _ = env.step(action)
        ppo_utils.append(reward)
        if done: break
    return ppo_utils

if __name__ == "__main__":
    model = train_ppo_agent()
    utils = test_ppo_agent(model)
    print("Sample PPO rewards:", utils[:10])
