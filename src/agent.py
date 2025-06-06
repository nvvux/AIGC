import random
from typing import Dict, List

class QLearningAgent:
    """Tabular Q-learning agent for discrete action spaces."""

    def __init__(self, actions: List[int], learning_rate=0.1, discount=0.9, epsilon=0.1):
        self.q_table: Dict[int, float] = {a: 0.0 for a in actions}
        self.actions = actions
        self.lr = learning_rate
        self.gamma = discount
        self.epsilon = epsilon

    def select_action(self) -> int:
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        return max(self.actions, key=lambda a: self.q_table[a])

    def update(self, action: int, reward: float):
        old_value = self.q_table[action]
        new_value = old_value + self.lr * (reward + self.gamma * 0 - old_value)
        self.q_table[action] = new_value
