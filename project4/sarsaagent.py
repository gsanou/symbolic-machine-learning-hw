from math import inf, ceil

from carddeck import *
from states import ValueAcesState, DealerValueAcesState, InitialState
from state_mappings import StateMapping, StateActionMapping
from blackjack import BlackjackObservation


class SarsaAgent:
    '''
    Here you will provide your implementation of SARSA method.
    You are supposed to implement train() method. If you want
    to, you can split the code in two phases - training and
    testing, but it is not a requirement.

    For SARSA explanation check AIMA book or Sutton and Burton
    book. You can choose any strategy and/or step-size function
    (learning rate).
    '''

    def __init__(self, env, number_of_epochs):
        self.env = env
        self.number_of_epochs = number_of_epochs

        self.utility = StateActionMapping(2)
        self.state_factory = DealerValueAcesState

        self.na_visits = StateActionMapping(2)
        self.n_visits = StateMapping()

        self.gamma = 0.9

        self.is_training = True

        self.observed_state = {i: [0] * self.number_of_epochs for i in range(4)}

    def train(self):
        for i in range(self.number_of_epochs):
            if i % 1000 == 0:
                print(i)

            observation = self.env.reset()  # type: BlackjackObservation
            terminal = False
            reward = 0
            state = self.state_factory.from_observation(observation, reward, terminal)
            action = self.make_step(state)

            while not terminal:
                self.n_visits[state] += 1
                self.na_visits[(state, action)] += 1

                observation, reward, terminal, _ = self.env.step(action)
                next_state = self.state_factory.from_observation(observation, reward, terminal)
                next_action = self.make_step(state)

                self.utility[(state, action)] = self.update_function(state, action, reward, next_state, next_action)

                state = next_state
                action = next_action

            self.observed_state[0][i] = self.utility[(self.state_factory(Rank.FOUR, 21, 0), 0)]
            self.observed_state[1][i] = self.utility[(self.state_factory(Rank.FOUR, 21, 0), 1)]
            self.observed_state[2][i] = self.utility[(self.state_factory(Rank.ACE, 5, 1), 0)]
            self.observed_state[3][i] = self.utility[(self.state_factory(Rank.ACE, 5, 1), 1)]

    def update_function(self, state, action, reward, next_state, next_action):
        return self.utility[(state, action)] + self.alpha(state, action) * (
                reward + self.gamma * self.utility[(next_state, next_action)] - self.utility[(state, action)]
        )

    def make_step(self, state):
        if self.is_training and self.epsilon(state) > self.env.env.np_random.rand():
            return self.env.action_space.sample()
        else:  # we go greedy
            best_utility = - inf
            best_action = None

            for action, utility in self.utility.get_action_utility(state).items():
                if utility > best_utility:
                    best_action = action
                    best_utility = utility

            return best_action

    def epsilon(self, state):
        c = 20
        return c / (c - 1 + self.n_visits[state])

    def alpha(self, state, action):
        c = 2
        return c / (c - 1 + self.na_visits[(state, action)])
