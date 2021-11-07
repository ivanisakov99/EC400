
import numpy as np
import matplotlib.pyplot as plt
import random

# discounting rate
gamma = 1 

#reward outside of a terminal state
rewardSize = -1 

#size of grid
gridSize = 4 

#two terminal states
terminationStates = [[0,0]] 

#four actions depending on where we want to move 
actions = [0,1,2,3] 

#this function returns what to add to the current state depending on the action
def actionVector(index):
    if index==0:
        return [-1,0] #move up
    if index==1:
        return [1,0] #move down
    if index==2:
        return [0,1] #move right
    if index==3:
        return [0,-1] #move left
    

#number of iterations 
numIter=15;



    #this function returns a reward of rewardSize each step unless you are in a terminal state
    #in that case, it returns a reward of zero
    #it returns  the next state and reward
def actionRewardFunction(initialPosition, action):

    
    
    #first check if we are in a termination state 
    #in that case, reward is zero and the position remains the same 
    
    if initialPosition in terminationStates:
        return initialPosition, 0
    
    #if we are not in a termination state, we returne the variable rewardSize 
    reward = rewardSize
    
    #calculcate next position 
    finalPosition = np.array(initialPosition) + np.array(actionVector(action))
    
    #now check if the next position brings you out of the grid
    #if so, the finalposition should be the same as the initial position
    
    if -1 in finalPosition or gridSize in finalPosition: 
        finalPosition = initialPosition
        
    return finalPosition, reward


#given a state, this function returns the neighbors of that state 
def neighbors(state):

    
    r = [] #start with an empty list. will append neighbors one by one to r
   
    #go through every action
    for action in actions:
        #check where you go by taking that action
        tempstate = np.array(state) + np.array(actionVector(action))
        #if you go outside the boundaries, don't add anything to r
        if -1 in tempstate or gridSize in tempstate:
            r=r
        #otherwise, add the destination to r    
        else:
                r.append(tempstate)
    #return the list r
    return r
   
    


#define the state space
states = [[i, j] for i in range(gridSize) for j in range(gridSize)]



#initialize value map to all zeros 
valueMap = np.zeros((gridSize, gridSize))

#will record thhe best action to take in a given state
bestAction = np.zeros((gridSize, gridSize))


#value iteration loop
for count in range(numIter):    
    


    #using notation from Sutton & Barto here 
    #deltas will be differences of current value iterate - next value iterate
    deltas = []

        
    #make a copy of the value function to manipulate during the algorithm
    copyValueMap = np.copy(valueMap)
    
    #this will be set to Vcurrent - Vnext
    deltaState = []
    
    
    for state in states:
    
        #the next variable will be equal to the new V iterate by the end of the process
        weightedRewards = []
        
        
        #Compute the Bellman iterate
        for action in actions:
            #compute next position and reward from taking that action
            finalPosition, reward = actionRewardFunction(state, action)
            
           
            
            weightedReward = reward+(gamma*valueMap[finalPosition[0], finalPosition[1]]) #note update uses valueMap, not copyValueMap
            
            #create a list of all the rewards we get
            weightedRewards.append(weightedReward)
            
        #take the action with theh best reward 
        copyValueMap[state[0],state[1]] = np.amax(weightedRewards)
        bestAction[state[0], state[1]] = np.argmax(weightedRewards)
        #append Vcurrent-Vnext for the current state
        deltaState.append(np.abs(copyValueMap[state[0], state[1]]-valueMap[state[0], state[1]]))
        
      
        
    #this is now an array of size numIterations, where every entry is an array of Vcurrent-Vmax    
    deltas.append(deltaState)
    
    #update the value map with what we just computed
    valueMap = copyValueMap
    


result = np.chararray((gridSize, gridSize))

#we now construct this character array from the policy
for i in range(gridSize):
    for j in range(gridSize):
        #compute action to take from [i,j]
        temp = bestAction[i,j]
       
       # Now display what that action is in words (i.e., U for up, D for down, etc)
        if temp == 0:
            tempchar = 'U'
        if temp == 1:
            tempchar = 'D'
        if temp == 2:
            tempchar = 'R'
        if temp == 3:
            tempchar = 'L'
        result[i,j]=tempchar

#print the final policy and the final value function
print("")
print("The Final Policy")    
print("")    
print(result.decode('utf-8'))



print("")
print("The Final Value Function")
print("")
print(valueMap)


#main issue I had writing this code: difficulty with data types
#
