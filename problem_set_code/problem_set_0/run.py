import numpy as np
import env
import policies as p
import matplotlib.pyplot as plt

N: int = 10000
T = 4
K = 4

table: dict = {
    1: [0, 0, 1, 1],
    2: [0, 0, 1, 1],
    3: [1, 1, 1, 1]
}

random: list = []
optimal: list = []

apt = env.ApartmentEnv(T, K)

p_random: p.RandomPolicy = p.RandomPolicy(T)
p_optimal: p.OptimalPolicy = p.OptimalPolicy(table)

def run(policy, track):
    for i in range(N):
        obs = apt.reset()[0]
        terminated = False
        
        for i in range(T):
            action: int = policy.act(obs)

            if terminated is True:
                break
            
            if action == 1:
                step = apt.step(action)
                track.append(step[1])
                terminated = step[2]

            else:
                step = apt.step(action)
                obs = step[0]
                terminated = step[2]

            if i == T - 1:
                track.append(0)

def run_t() -> np.array:
    u_min: list = [1, 2, 3, 4]
    threshold: list = []
    max_mean: float = 0
    max_array: np.array = None

    for i in range(len(u_min)):
        p_threshold: p.ThresholdPolicy = p.ThresholdPolicy(u_min[i])

        run(p_threshold, threshold)

        arr: np.array = np.array(threshold)

        if arr.mean() > max_mean:
            max_mean = arr.mean()
            max_array = arr

        threshold = []

    return max_array

def reject_all(rewards: np.array) -> float:
    reject_all: int = 0

    for i in range(len(rewards)):
        if rewards[i] == 0:
            reject_all += 1

    return reject_all / N

run(p_random, random)
run(p_optimal, optimal)

r_array: np.array = np.array(random)
t_array: np.array = run_t()
o_array: np.array = np.array(optimal)

r_mean: float = r_array.mean()
r_se: float = np.std(r_array, ddof=1) / np.sqrt(N)
r_rej: float = reject_all(r_array)

t_mean: float = t_array.mean()
t_se:float = np.std(t_array, ddof=1) / np.sqrt(N)
t_rej: float = reject_all(t_array)

o_mean: float = o_array.mean()
o_se: float = np.std(o_array, ddof=1) / np.sqrt(N)
o_rej: float = reject_all(o_array)

print(f"\nRandom Policy Mean Reward: {round(r_mean,3)}, Standard Error: {round(r_se, 3)}")
print(f"Threshold Policy Mean Reward: {round(t_mean, 3)}, Standard Error: {round(t_se, 3)}")
print(f"Optimal Policy Mean Reward: {round(o_mean, 3)}, Standard Error: {round(o_se, 3)}\n")

print(f"The Random Policy rejected all apartments in {round(r_rej * 100, 2)}% of searches")
print(f"The Threshold Policy rejected all apartments in {round(t_rej * 100, 2)}% of searches")
print(f"The Optimal Policy rejected all apartments in {round(o_rej * 100, 5)}% of searches")

plt.figure()
plt.hist(r_array, label="Random Policy")
plt.hist(t_array, label="Threshold Policy")
plt.hist(o_array, label="Optimal Policy")
plt.xlabel("Reward")
plt.ylabel("Count")
plt.legend()
plt.show()