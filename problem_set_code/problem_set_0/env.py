"""
"""
from gymnasium import Env, spaces
import numpy as np

class ApartmentEnv(Env):
    
    def __init__(self, T: int, K: int, seed=None):
        super().__init__()
        self.T = T
        self.K = K
        self.t = None
        self.utils = None
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low=np.array([1, 1]), high=np.array([T, K]), shape=(2, ), dtype=np.float32)

    def reset(self, seed=None, options=None) -> tuple:
        self.t = 1
        self.utils = np.random.randint(1, self.K + 1, size=self.T)

        return (self.t, self.utils[self.t - 1]), {}

    def step(self, action: int) -> tuple:
        terminated: bool = False
        u_t = self.t - 1
        reward: int = 0

        if self.t >= self.T:
            terminated = True

        if action == 1:
            terminated = True
            reward = self.utils[u_t]

        if self.t < self.T:
            self.t = self.t + 1


        return (self.t, self.utils[u_t]), reward, terminated, False, {}