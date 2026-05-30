import numpy as np

class RandomPolicy:
    def __init__(self, T: int):
        self.T = T

    def act(self, obs: tuple) -> int:
        return np.random.random() < 1 / self.T

class ThresholdPolicy:
    def __init__(self, u_min: int):
        self.u_min = u_min

    def act(self, obs: tuple) -> int:
        if obs[1] >= self.u_min:
            return 1

        else:
            return 0

class OptimalPolicy:
    def __init__(self, table: dict):
        self.table = table

    def act(self, obs: tuple) -> int:
        return self.table[obs[0]][obs[1] - 1]