from math import sqrt

from numpy import NaN

from blackjack import BlackjackObservation
from carddeck import *
from states import ValueAcesState, InitialState
from state_mappings import StateMapping


class TDAgent:
    '''
    Implementaion of an agent that plays the same strategy as the dealer.
    This means that the agent draws a card when sum of cards in his hand
    is less than 17.

    Your goal is to modify train() method to learn the state utility function.
    I.e. you need to change this agent to a passive reinforcement learning
    agent that learns utility estimates using temporal diffrence method.
    '''

    def __init__(self, env, number_of_epochs):
        self.env = env
        self.number_of_epochs = number_of_epochs

        self.utility = StateMapping()
        self.state_factory = ValueAcesState

        self.n_visits = StateMapping()
        self.gamma = 0.95

        self.observed_state = [0] * self.number_of_epochs

    def train(self):
        for i in range(self.number_of_epochs):
            print(i)
            observation = self.env.reset()  # type: BlackjackObservation
            terminal = False
            reward = 0
            state = self.state_factory.from_observation(observation, reward, terminal)

            while not terminal:
                self.n_visits[state] += 1

                action = self.make_step(observation, reward, terminal)
                observation, reward, terminal, _ = self.env.step(action)

                next_state = self.state_factory.from_observation(observation, reward, terminal)

                self.utility[state] = self.update_function(state, reward, next_state)
                state = next_state

            self.observed_state[i] = self.utility[self.state_factory(20, 0)]

    def update_function(self, state, reward, next_state):
        return self.utility[state] + self.alpha(state) * (
                reward + self.gamma * self.utility[next_state] - self.utility[state]
        )

    def alpha(self, state):
        c = 10
        return c / (c - 1 + self.n_visits[state])

    # def alpha(self, state):
    #     return 1 / sqrt(self.n_visits[state])

    def make_step(self, observation, reward, terminal):
        return 1 if observation.player_hand.value() < 17 else 0
