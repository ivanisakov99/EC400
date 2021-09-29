import numpy as np
import matplotlib.pyplot as plt
import random

# Discounting rate
gamma = 0.99

# Reward outside of a terminal state
rewardSize = -1 

# Size of grid
gridSize = 4 

# Two terminal states
terminationStates = [[0,0], [gridSize-1, gridSize-1]] 

# Four actions depending on where we want to move 
actions = [[-1, 0], [1, 0], [0, 1], [0, -1]] 

# Number of iterations we want to do
numIterations = 100


def actionRewardFunction(initialPosition, action):
    # This function returns a reward of rewardSize each step unless you are in a terminal state
    # in that case, it returns a reward of zero
    # it returns the next state and reward
    
    
    # First check if we are in a termination state 
    # in that case, reward is zero and the position remains the same 
    
    if initialPosition in terminationStates:
        return initialPosition, 0
    
    # If we are not in a termination state, we return the variable rewardSize 
    reward = rewardSize
    
    # Calculcate next position 
    finalPosition = np.array(initialPosition) + np.array(action)
    
    # Now check if the next position brings you out of the grid
    # if so, the finalposition should be the same as the initial position
    
    if -1 in finalPosition or gridSize in finalPosition: 
        finalPosition = initialPosition
        
    return finalPosition, reward
    
    
def defineAction(action):
    if(action == actions[0]):
        return 'U'
    elif(action == actions[1]):
        return 'D'
    elif(action == actions[2]):
        return 'R'
    elif(action == actions[3]):
        return 'L'
    else:
        return '-'


# Initialise value map to all zeros 
valueMap = np.zeros((gridSize, gridSize))

# Policy Map
policyMap = np.full((gridSize, gridSize), '-')

# Define the state space
states = [[i, j] for i in range(gridSize) for j in range(gridSize)]

deltas = []
for it in range(numIterations):
    
    # Make a copy of the value function to manipulate during the algorithm
    copyValueMap = np.copy(valueMap)
    temp = np.zeros((1,4))
    i = 0

    # This will be set to Vcurrent - Vnext
    deltaState = []
    for state in states:
    
        # The next variable will be equal to the new V iterate by the end of the process
        weightedRewards = 0
        temp = np.zeros((1, 4))
        i = 0

        # Compute the Bellman iterate
        for action in actions:
            # Compute next position and reward from taking that action
            finalPosition, reward = actionRewardFunction(state, action)
            # Updating weightedRewards => V(s) <- ∑ π(a|s) * ∑ P(s'| s, a) * (r + y(V(s')))
            weightedRewards += 1/(len(actions)) * (reward + gamma * valueMap[finalPosition[0], finalPosition[1]])
            temp[0][i] = 1/(len(actions)) * (reward + gamma * valueMap[finalPosition[0], finalPosition[1]])
            i = i + 1

        i = np.argmax(temp)
        policyMap[state[0], state[1]] = (defineAction(actions[i]), '-')[weightedRewards == 0]
            
        
        # Append Vcurrent-Vnext for the current state
        deltaState.append(np.abs(copyValueMap[state[0], state[1]]-weightedRewards))
        
        # Update the value of the next state, but in the copy rather than the original
        copyValueMap[state[0], state[1]] = weightedRewards
        
    # This is now an array of size numIterations, where every entry is an array of Vcurrent-Vmax    
    deltas.append(deltaState)
    
    # Update the value map with what we just computed
    valueMap = copyValueMap
    
    # For selected iterations, print the value function
    if it in [0,1,2,9, 99, numIterations-1]:
        print("Iteration {}".format(it+1))
        print(valueMap)
        print("")


# Print Policy Map
print("Policy Map")
print(policyMap)

# Plot how the deltas decay        
plt.figure(figsize=(20, 10))
plt.plot(deltas)
# plt.show()
# print(deltas)

