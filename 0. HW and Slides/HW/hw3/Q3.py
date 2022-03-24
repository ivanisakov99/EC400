from os import stat
import numpy as np

# Time
time = 1
# Discount Factor
gamma = 0.5
# Q-Values array
qValues = np.array([[16, 16], [0, 16]])
# Action transitions
actionsToNextState = np.array([0, 1, 1])
# Rewards associated with the actions
rewards = np.array([[1, 0], [0, 2]])

sequence = [[0, 0], [0, 1], [1, 1]]

while(1):
    prev = qValues.copy()
    for state, action in sequence:
        # Previous matrix
        # Alpha
        alpha = 1 / time
        
        # The Q-Value at the next state
        nextState = actionsToNextState[action]
        nextQvalue = max(qValues[nextState])

        # Reward corresponding to the action
        reward = rewards[state][action]

        # Q_t+1(s_t, a_t) = Q_t(s_t, a_t) + alpha_t * (r(s_t, a_t) + y * max(Q_t(s_t+1, a)) - Q_t(s_t, a_t))
        qValues[state][action] = qValues[state][action] + alpha * \
            (reward + gamma * nextQvalue - qValues[state][action])

        # Time step increase
        time += 1

    # If it converged, break
    if(np.array_equal(prev, qValues)):
        break


print(qValues)
