class StackelbergEnv:
    """Simple Stackelberg game environment for AIGC services."""

    def __init__(self, price_levels=6, demand_levels=6, cost_factor=0.1, value_factor=1.5):
        self.price_levels = list(range(price_levels))
        self.demand_levels = list(range(demand_levels))
        self.cost_factor = cost_factor
        self.value_factor = value_factor

    def follower_best_response(self, price: int) -> int:
        """Compute follower demand that maximizes its utility given a price."""
        best_demand = 0
        best_utility = float('-inf')
        for d in self.demand_levels:
            utility = self.value_factor * d - 0.5 * d * d - price * d
            if utility > best_utility:
                best_utility = utility
                best_demand = d
        return best_demand

    def step(self, price_action: int) -> tuple:
        """Perform one game step for a leader price action."""
        if price_action not in self.price_levels:
            raise ValueError('Invalid price action')
        demand = self.follower_best_response(price_action)
        reward = price_action * demand - self.cost_factor * demand * demand
        return reward, demand
