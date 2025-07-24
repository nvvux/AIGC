# train_ppo.py

from ppo_env import StackelbergGymEnv
from tppo_policy import TransformerActorCriticPolicy
from stable_baselines3 import PPO

def train_ppo_agent(
    num_vmu=10,
    cost_per_step=1,
    p_min=1,
    p_max=20,
    history_len=10,
    features_dim=64,
    nhead=4,
    nlayer=2,
    total_timesteps=200_000,
):
    env = StackelbergGymEnv(
        num_vmu=num_vmu,
        history_len=history_len,
        cost_per_step=cost_per_step,
        p_min=p_min,
        p_max=p_max
    )
    policy_kwargs = dict(
        seq_len=history_len,
        features_dim=features_dim,
        nhead=nhead,
        nlayer=nlayer,
    )
    model = PPO(
        TransformerActorCriticPolicy,
        env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        #tensorboard_log="./ppo_tb/"
    )
    model.learn(total_timesteps=total_timesteps)
    model.save("ppo_stackelberg_tppo")
    return model

def test_ppo_agent(model, num_vmu=10, cost_per_step=1, p_min=1, p_max=20, history_len=10):
    env = StackelbergGymEnv(
        num_vmu=num_vmu,
        history_len=history_len,
        cost_per_step=cost_per_step,
        p_min=p_min,
        p_max=p_max
    )
    obs = env.reset()
    ppo_utils = []
    prices = []
    participants = []
    for i in range(600):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        # Lưu lại thông tin reward, price, participants nếu muốn vẽ lại Fig. 2
        ppo_utils.append(reward)
        prices.append(action[0])
        participants.append(obs[-1])  # số participants bước hiện tại
        if done:
            break
    return dict(rewards=ppo_utils, prices=prices, participants=participants)

if __name__ == "__main__":
    model = train_ppo_agent()
    results = test_ppo_agent(model)
    print("Sample PPO rewards:", results["rewards"][:10])
    print("Sample prices:", results["prices"][:10])
    print("Sample participants:", results["participants"][:10])
