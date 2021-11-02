# qlearningAgents.py
# ------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random
import util
import math


class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        self.Qvalues = util.Counter()
        # you should be able to address self.Qvalues[(state,action)]
        ReinforcementAgent.__init__(self, **args)

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        if (state, action) in self.Qvalues:
            return self.Qvalues[(state, action)]
        else:
            return 0.0

    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        values = [
            self.getQValue(state, action)
            for action in self.getLegalActions(state)
        ]
        if len(values):
            return max(values)
        else:
            return 0.0

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        value = self.getValue(state)
        actions = [
            action for action in self.getLegalActions(state) \
            if (self.getQValue(state, action) == value)
        ]

        if len(actions):
            return random.choice(actions)
        else:
            return None

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        legalActions = self.getLegalActions(state)
        action = None

        if util.flipCoin(self.epsilon):
            action = random.choice(legalActions)
        else:
            action = self.getPolicy(state)

        return action

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        gamma = self.discount
        alpha = self.alpha
        qValue = self.getQValue(state, action)
        nextValue = self.getValue(nextState)

        newValue = qValue + alpha * (reward + gamma * nextValue - qValue)

        self.Qvalues[(state, action)] = newValue
        

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self,
                 epsilon=0.05,
                 gamma=0.8,
                 alpha=0.2,
                 numTraining=0,
                 **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self, state)
        self.doAction(state, action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector,
          where * is the dotProduct operator
        """
        # Get features
        features = self.featExtractor.getFeatures(state, action)
        ans = 0

        # Dot product between the weights and the features
        for feature in features:
            ans += self.weights[feature] * features[feature]
        return ans

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        # Get features
        features = self.featExtractor.getFeatures(state, action)
        
        # ??? getValue is a pure calculation of Q-values, rather than the approximation
        nudge = (reward + self.discount * self.getValue(nextState)) - \
            self.getQValue(state, action)

        # w_(t + 1) = w_t + alpha * ((reward_t + gamma * max_a(w_t * x(s_t', a))) - w_t) * x(s_t, a_t)
        for feature in features:
            self.weights[feature] = self.weights[feature] + self.alpha * nudge * features[feature]

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            print("Weights:")
            print(self.weights)
            pass
