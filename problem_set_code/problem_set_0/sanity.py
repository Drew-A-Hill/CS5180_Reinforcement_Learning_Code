import env
import numpy as np

apt = env.ApartmentEnv(10, 10)
reset = apt.reset()
print(f"t: {reset[0][0]}\tU_t: {reset[0][1]}\t")

for i in range(apt.T):
    action = np.random.randint(0, 2)
    episode = apt.step(action)

    print(f"t: {episode[0][0]}\tU_t: {episode[0][1]}\t action: {action}\t reward: {episode[1]}\t terminated: {episode[2]}\n")

    if episode[2] is True:
        break