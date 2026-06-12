import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

def value_iteration(P, gamma, theta):
    states_n = len(P)
    actions_n = len(P[0])

    V = np.zeros(states_n)

    iteration_count = 0

    while True:
        v_new = np.zeros(states_n)

        for states in range(states_n):
            q_vals = []

            for actions in range(actions_n):
                q = 0
            
                for (prob, next_state, reward, terminated) in P[states][actions]:
                    q += prob * (reward + gamma * V[next_state])

                q_vals.append(q)

            v_new[states] = max(q_vals)

        t = np.max(np.abs(v_new - V))

        V = v_new

        iteration_count += 1

        if t < theta * ((1 - gamma) / gamma):
            break

    policy = np.zeros(states_n, dtype=int)

    for states in range(states_n):
        q_vals = []

        for actions in range(actions_n):
            q = 0

            for (prob, next_state, reward, terminated) in P[states][actions]:
                q += prob * (reward + gamma * V[next_state])

            q_vals.append(q)
        
        policy[states] = np.argmax(q_vals)

    return V, policy, iteration_count

def show_policy(policy, table_size):
    arrows = ["←", "↓", "→", "↑"]

    holes = [5, 7, 11, 12]
    goal = [15]

    print("\nPolicy Table")
    print("+-------------+")
    for states in range(table_size * table_size):
        if states in holes:
            symbol = " H "
        
        elif states in goal:
            symbol = " G "

        else:
            symbol = f" {arrows[policy[states]]} "

        if states % table_size == 0:
            print("|", end="")
        
        print(symbol, end="")

        if states % table_size == table_size - 1:
            print("|")

    print("+-------------+")

def show_values(V, table_size):
    print("\nV* Table")

    row = []
    for i in range(table_size * table_size):
        row.append(float(round(V[i], 4)))
        if (i + 1) % table_size == 0:
            print(row)
            row = []

def main():
    env = gym.make("FrozenLake-v1", is_slippery=True)
    P = env.unwrapped.P

    gamma = 0.99
    theta = 1e-4

    V, policy, iterations = value_iteration(P, gamma, theta)

    print(f"\nConverges after {iterations} iterations")
    show_policy(policy, 4)
    show_values(V, 4)

if __name__ == "__main__":
    main()