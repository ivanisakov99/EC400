# <ins>HW2: Q-Learning</ins>
Write code to do Q-learning on gridworld by editing the `qlearningAgents.py` file. In that file, fill in the code for the functions `getQValue`, `ComputeValuefromQvalues`, `computeActionFromQValues`, `getAction`, and `update`.

Make sure that in your computeValueFromQValues and computeActionFromQValues functions, you only access Q values by calling getQValue.

Once you fill in your own code, remove the util.raiseNotDefined() commands.
Once you are done, you can run 

    python3 gridworld.py -a q -k 15

to see the outcome of 15 episodes of Q-learning.