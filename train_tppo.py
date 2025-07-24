# train_tppo.py
from ppo_env import StackelbergGymEnv
from stable_baselines3 import PPO
from tppo_policy import TransformerActorCriticPolicy

def train_tppo_agent(
    num_vmu=10,
    cost_per_step=1,
    p_min=1,
    p_max=20,
    seq_len=10,
    features_dim=64,
    nhead=4,
    nlayer=2,
    total_timesteps=200_000,
):
    env = StackelbergGymEnv(
        num_vmu=num_vmu,
        history_len=seq_len,
        cost_per_step=cost_per_step,
        p_min=p_min,
        p_max=p_max
    )
    policy_kwargs = dict(
        seq_len=seq_len,
        features_dim=features_dim,
        nhead=nhead,
        nlayer=nlayer
    )
    model = PPO(
        TransformerActorCriticPolicy,
        env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        #tensorboard_log="./tppo_tb/"
    )
    model.learn(total_timesteps=total_timesteps)
    model.save("tppo_stackelberg")
    return model

def test_tppo_agent(model, num_vmu=10, cost_per_step=1, p_min=1, p_max=20, seq_len=10):
    env = StackelbergGymEnv(
        num_vmu=num_vmu,
        history_len=seq_len,
        cost_per_step=cost_per_step,
        p_min=p_min,
        p_max=p_max
    )
    obs = env.reset()
    tppo_utils = []
    prices = []
    participants = []
    for i in range(600):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        tppo_utils.append(reward)
        prices.append(action[0])
        participants.append(obs[-1])
        if done: break
    return dict(rewards=tppo_utils, prices=prices, participants=participants)

if __name__ == "__main__":
    model = train_tppo_agent()
    results = test_tppo_agent(model)
    print("Sample TPPO rewards:", results["rewards"][:10])
    print("Sample prices:", results["prices"][:10])
    print("Sample participants:", results["participants"][:10])
