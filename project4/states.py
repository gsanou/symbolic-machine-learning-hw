# coding=utf-8
from abc import ABC, abstractmethod
from numbers import Number

from blackjack import BlackjackObservation
from carddeck import *


class State(ABC):
    @abstractmethod
    def __hash__(self):
        pass

    @abstractmethod
    def __eq__(self, other):
        pass

    @classmethod
    @abstractmethod
    def from_observation(cls, observation: BlackjackObservation, reward: Number, terminal: bool) -> "State":
        pass


class InitialState(State):
    def __hash__(self):
        return hash(0)

    def __eq__(self, other):
        if not isinstance(other, InitialState):
            return False
        return True

    def __str__(self):
        return "<Init>"

    def __lt__(self, other):
        return False

    @classmethod
    def from_observation(cls, observation: BlackjackObservation, reward: Number, terminal: bool) -> "State":
        return cls()


class TerminalState(State):
    def __init__(self, reward: Number):
        self.reward = reward

    def __hash__(self):
        return hash(self.reward)

    def __eq__(self, other):
        if not isinstance(other, TerminalState):
            return False
        return self.reward == other.reward

    def __str__(self):
        return "<T%d>" % self.reward

    def __lt__(self, other):
        return False

    @classmethod
    def from_observation(cls, observation: BlackjackObservation, reward: Number, terminal: bool) -> "State":
        return cls(reward)


class ValueAcesState(State):
    def __init__(self, non_aces_value, num_of_aces):
        self.non_aces_value = non_aces_value
        self.num_of_aces = num_of_aces

    def __hash__(self):
        return hash(hash(self.non_aces_value) + hash(self.num_of_aces))

    def __eq__(self, other):
        if not isinstance(other, ValueAcesState):
            return False
        return self.non_aces_value == other.non_aces_value and self.num_of_aces == other.num_of_aces

    def __repr__(self):
        return "<S%d, %d>" % (self.non_aces_value, self.num_of_aces)

    def __lt__(self, other):
        if not isinstance(other, ValueAcesState):
            return True
        if self.non_aces_value < other.non_aces_value:
            return True
        elif self.non_aces_value == other.non_aces_value:
            return self.num_of_aces < other.num_of_aces
        return False

    @classmethod
    def from_observation(cls, observation: BlackjackObservation, reward: Number, terminal: bool):
        if terminal:
            return TerminalState.from_observation(observation, reward, terminal)

        num_of_aces, non_aces_value = 0, 0
        for card in observation.player_hand.cards:
            if card.rank == Rank.ACE:
                num_of_aces += 1
            else:
                non_aces_value += card.value()

        return ValueAcesState(non_aces_value, num_of_aces)


class DealerValueAcesState(State):
    def __init__(self, dealer_card: Rank, non_aces_value, num_of_aces):
        self.dealer_card = self.transform_card(dealer_card)
        self.non_aces_value = non_aces_value
        self.num_of_aces = num_of_aces

    def __hash__(self):
        return hash(hash(self.dealer_card) + hash(self.non_aces_value) + hash(self.num_of_aces))

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return False
        return self.non_aces_value == other.non_aces_value \
               and self.num_of_aces == other.num_of_aces \
               and self.dealer_card == other.dealer_card

    def __repr__(self):
        return "<D%s, S%d, %d>" % (self.dealer_card, self.non_aces_value, self.num_of_aces)

    def __lt__(self, other):
        if not isinstance(other, DealerValueAcesState):
            return True

        if self.dealer_card.value < other.dealer_card.value:
            return True
        elif self.dealer_card == other.dealer_card:
            if self.non_aces_value < other.non_aces_value:
                return True
            elif self.non_aces_value == other.non_aces_value:
                return self.num_of_aces < other.num_of_aces
        return False

    @classmethod
    def from_observation(cls, observation: BlackjackObservation, reward: Number, terminal: bool):
        if terminal:
            return TerminalState.from_observation(observation, reward, terminal)

        num_of_aces, non_aces_value = 0, 0
        for card in observation.player_hand.cards:
            if card.rank == Rank.ACE:
                num_of_aces += 1
            else:
                non_aces_value += card.value()

        return cls(observation.dealer_hand.cards[0].rank, non_aces_value, num_of_aces)

    def transform_card(self, dealer_card):
        if dealer_card in {Rank.JACK, Rank.KING, Rank.QUEEN, Rank.TEN}:
            return Rank.TEN
        else:
            return dealer_card
