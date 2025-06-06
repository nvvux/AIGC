from environment import StackelbergEnv
from agent import QLearningAgent


def train(episodes: int = 1000):
    env = StackelbergEnv()
    agent = QLearningAgent(actions=env.price_levels)
    history = []
    for ep in range(episodes):
        action = agent.select_action()
        reward, demand = env.step(action)
        agent.update(action, reward)
        history.append((action, demand, reward))
    return history, agent


if __name__ == "__main__":
    history, agent = train(episodes=200)
    print("Trained Q-values:")
    for action, q in sorted(agent.q_table.items()):
        print(f"Price {action}: Q={q:.2f}")
