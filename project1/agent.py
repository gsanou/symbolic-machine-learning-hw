from typing import Iterable

import numpy as np
from math import factorial

from tutorial3 import Disjunction
from itertools import combinations, chain


def cnk(n, k):
    return factorial(n) // factorial(k) // factorial(n - k)


VERBOSE = False


class Hypothesis:
    def __init__(self, n_variables, s):
        self.clauses = self.generate_all_disjunctions(n_variables, s)
        self.incorrect_clauses = None

    @staticmethod
    def set_cuts(l):
        source_set = set(l)
        subsets = []
        for subset in chain.from_iterable(combinations(l, n) for n in range(len(l) + 1)):
            subsets.append((list(source_set.difference(subset)), list(subset)))
        return subsets

    # Version 1: too many clauses generated (redundant)
    # @staticmethod
    # def generate_all_disjunctions(n_variables, s) -> Iterable[Disjunction]:
    #     disjunctions = set()
    #
    #     for i in range(1, s + 1):
    #         for indices in combinations(range(2 * n_variables), i):
    #             flags = np.zeros((2 * n_variables), dtype=np.bool)
    #             flags[list(indices)] = True
    #
    #             positive_literals = flags[:n_variables]
    #             negative_literals = flags[n_variables:]
    #             disjunctions.add(Disjunction(positive_literals, negative_literals))
    #
    #     return disjunctions

    @staticmethod
    def generate_all_disjunctions(n_variables, s) -> Iterable[Disjunction]:
        disjunctions = set()

        for i in range(1, s + 1):
            literal_combinations = combinations(range(n_variables), i)
            literal_divisions = Hypothesis.set_cuts(range(i))

            for chosen_literals_indices in literal_combinations:
                chosen_literals_indices = np.array(chosen_literals_indices)

                for positive_indices, negative_indices in literal_divisions:
                    positive_literals = np.zeros(n_variables, dtype=np.bool)
                    negative_literals = np.zeros(n_variables, dtype=np.bool)

                    positive_literals[chosen_literals_indices[positive_indices]] = True
                    negative_literals[chosen_literals_indices[negative_indices]] = True

                    disjunctions.add(Disjunction(positive_literals, negative_literals))

        return disjunctions

    def update(self, interpretation):
        self.clauses = set(filter(lambda c: c.evaluate(interpretation), self.clauses))

    def evaluate(self, interpretation):
        for clause in self.clauses:
            if not clause.evaluate(interpretation):
                return False
        return True

    def __str__(self):
        return " & ".join("(%s)" % str(clause) for clause in self.clauses)


class Agent:
    def __init__(self, epsilon, delta):
        self.epsilon = epsilon
        self.delta = delta

        # init in constructor
        self.previous_interpretation = None
        self.n_variables = None
        self.hypothesis = None
        self.previous_estimate = None
        self.s = None

    def compute_required_training_dataset_size(self):
        n = 0
        for i in range(1, self.s + 1):
            n += cnk(self.n_variables, i) * (2 ** i)

        m = int(
            (1 / self.epsilon) * (n * np.log(2) - np.log(self.delta))
        )
        return m

    def process_first_observation(self, interpretation):
        self.previous_interpretation = interpretation
        self.previous_estimate = self.hypothesis.evaluate(interpretation)
        return self.previous_estimate

    def predict(self, interpretation, reward):
        if reward is not None and reward == 0 and self.previous_estimate is False:
            self.hypothesis.update(self.previous_interpretation)

        self.previous_interpretation = interpretation
        self.previous_estimate = self.hypothesis.evaluate(interpretation)
        return self.previous_estimate

    def interact_with_oracle(self, oracle_session):
        self.n_variables, self.s = oracle_session.request_parameters()
        self.hypothesis = Hypothesis(self.n_variables, self.s)

        m = self.compute_required_training_dataset_size()
        first_sample = oracle_session.request_dataset(m)
        first_sample = first_sample.reshape((1, -1))
        prediction = self.process_first_observation(first_sample)

        while oracle_session.has_more_samples():
            interpretation, reward = oracle_session.predict(prediction)
            interpretation = interpretation.reshape((1, -1))
            prediction = self.predict(interpretation, reward)
