from ppo_env import StackelbergGymEnv
from stable_baselines3 import PPO
from tppo_policy import TransformerActorCriticPolicy

def train_tppo_agent(num_vmu=10, cost_per_step=1, p_max=20, seq_len=10):
    env = StackelbergGymEnv(num_vmu=num_vmu, history_len=seq_len, cost_per_step=cost_per_step, p_max=p_max)
    model = PPO(
        policy=TransformerActorCriticPolicy,
        env=env,
        policy_kwargs=dict(seq_len=seq_len, features_dim=64, nhead=4, nlayer=2),
        verbose=1
    )
    model.learn(total_timesteps=200_000)
    model.save("tppo_stackelberg")
    return model

def test_tppo_agent(model, num_vmu=10, cost_per_step=1, p_max=20, seq_len=10):
    env = StackelbergGymEnv(num_vmu=num_vmu, history_len=seq_len, cost_per_step=cost_per_step, p_max=p_max)
    obs = env.reset()
    tppo_utils = []
    for i in range(600):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _ = env.step(action)
        tppo_utils.append(reward)
        if done: break
    return tppo_utils
if __name__ == "__main__":
    model = train_tppo_agent()
    utils = test_tppo_agent(model)
    print("Sample TPPO rewards:", utils[:10])