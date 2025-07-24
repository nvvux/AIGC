# train.py
import torch
import numpy as np
from agent import TDRLAgent
from env import AIGCEnv


def train_agent(epochs=1000, batch_size=32, explore_prob=0.1, verbose=True):
    """
    Huấn luyện tác tử ASP để học chính sách chọn số bước khuếch tán tối ưu (S*)
    sao cho tối đa hóa tiện ích ASP (Eq. 6: U_p = ∑_k(p - c)·S_k)

    Trả về:
    - trained_agent: tác tử đã học xong
    - reward_log: danh sách tiện ích ASP qua các epoch
    """
    env = AIGCEnv()
    agent = TDRLAgent(env)
    reward_log = []

    for epoch in range(epochs):
        state = env.reset()
        action = agent.select_action(state, explore_prob)
        next_state, reward, _, _ = env.step(action)

        agent.store_transition(state, action, reward, next_state)
        agent.train(batch_size=batch_size)
        reward_log.append(reward)

        if verbose and (epoch + 1) % 100 == 0:
            avg_reward = np.mean(reward_log[-100:])
            print(f"Epoch {epoch + 1}/{epochs} | ASP reward (avg last 100): {avg_reward:.4f}")

    return agent, reward_log
