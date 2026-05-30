import env
import numpy as np

t = env.ApartmentEnv(10, 10)
reset = t.reset()

for i in range(10):
    action = np.random.randint(0, 2)
    episode = t.step(action)

    print(f"t: {i + 1}\tU_t: {episode[0][1]}\t action: {action}\t reward: {episode[1]}\t terminated: {episode[2]}\n")

    if episode[2] is True:
        break