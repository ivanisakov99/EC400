import numpy as np
import math


#initialize Q to be the vector [16,16,16]
Q = 16*np.ones(3) 

#set the rewards vector 
rewards = np.array([1,0,2])

#discount factor
gamma = 1/2

#number of Q-learning updates we will do
num_iter = 500


#Main loop. As discussed in class, we'll simply cycle through the three state-action pairs provided in the example 
#this will be done by looking at the remainder when i is divided by 3, which is i % 3 in python notation 


for i in range(1,num_iter):

  
  if i % 3 == 0:
    Q[0] = Q[0] + (1/math.sqrt(i))*(rewards[0] + gamma*max(Q[0], Q[1]) - Q[0] ) #after  the state-action pair  s=1,a=1, the next state is 1, so we have to take the maximum of the two Q-values at state 1

  if i % 3 == 1:
    Q[1] = Q[1] + (1/math.sqrt(i))*(rewards[1] + gamma*Q[2] - Q[1]) #after  the station-action pair  s=1, a=2, the next state is 2, and there is only one action there

  if i % 3 == 2: 
    Q[2] = Q[2] + (1/math.sqrt(i))*(rewards[2] + gamma*Q[2] - Q[2]) #after  the state-action pair s=2,a=1, the next state is 2, and there is only one action there. 

print(Q)

