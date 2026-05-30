import numpy as np
import env

N: int = 10000
T: int = 4
K: int = 4
pi_random: np.array = np.array()
pi_threshold: np.array = np.array()
pi_opitmal: np.array = np.array()

for i in range(N):
    env = env.ApartmentEnv(T, K)
    env.reset()

    for i in range(T):
        
        env.step

