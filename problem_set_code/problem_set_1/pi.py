import gymnasium as gym
import numpy as np

def policy_iteration(P, gamma):
    states_n = len(P)
    actions_n = len(P[0])

    policy = np.zeros(states_n, dtype=int)

    iteration_count = 0

    while True:
        P_pi = np.zeros((states_n, states_n))
        R_pi = np.zeros(states_n)

        for states in range(states_n):
            for (prob, state_next, reward, terminated) in P[states][policy[states]]:
                P_pi[states][state_next] += prob
                R_pi[states] += prob * reward

        A = np.eye(states_n) - gamma * P_pi
        V = np.linalg.solve(A, R_pi)

        policy_new = np.zeros(states_n, dtype=int)

        for states in range(states_n):
            q_vals = []

            for actions in range(actions_n):
                q = 0

                for (prob, next_state, reward, terminated) in P[states][actions]:
                    q += prob * (reward + gamma * V[next_state])
                
                q_vals.append(q)

            policy_new[states] = np.argmax(q_vals)

        iteration_count += 1

        if np.array_equal(policy_new, policy):
            break

        policy = policy_new

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
        
def policy_compare(vi, pi):
    if np.array_equal(vi, pi):
        print("\nPolicies are the same\n")

    else:
        print("\nPolicies differ\n")

def main():
    env = gym.make("FrozenLake-v1", is_slippery=True)
    P = env.unwrapped.P

    gamma = 0.99

    V, policy, iterations = policy_iteration(P, gamma)

    print(f"\nConverges after {iterations} iterations")
    show_policy(policy, 4)
    show_values(V, 4)

if __name__ == "__main__":
    main()