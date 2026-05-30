import numpy as np

class RandomPolicy:
    def act(obs: tuple(int)) -> int:
        return np.random.randint(0, 2)

class ThresholdPolicy:
    def __init__(self, u_min: int):
        self.u_min = u_min

    def act(obs: tuple(int)) -> int:
        if obs[1][obs[0]] >= self.u_min:
            return 1

        else:
            return 0

class OptimalPolicy:
    def wt(utils: np.array) -> int:
        sum: int = 0

        for i in range(len(utils)):
            sum = sum + utils[i]

        return sum / len(utils)

    def update_v(T: int, utils: np.array):
        wt_1 = None
        wt 

    def act(obs: tuple(int)) -> int:
        pass




        