import matplotlib
import pandas as pd
from sarsaagent import SarsaAgent
from states import DealerValueAcesState

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def print_policy(agent):
    if not isinstance(agent, SarsaAgent):
        return

    d = {}

    for state in agent.utility.d.keys():
        actions = agent.utility.get_action_utility(state)

        if type(state) is not DealerValueAcesState:
            continue

        if state.num_of_aces > 2:
            continue

        if state.dealer_card not in d:
            d[state.dealer_card] = {}

        if actions[1] >= actions[0]:
            d[state.dealer_card][(state.non_aces_value, state.num_of_aces)] = "HIT"
        else:
            d[state.dealer_card][(state.non_aces_value, state.num_of_aces)] = "STAND"

    df = pd.DataFrame.from_dict(d)
    df.fillna("-", inplace=True)
    df.to_csv("strategy.csv")


def evaluate(rewards, agent=None):
    # TODO implement your own code here if you want to
    # or alternatively you can modify the existing code

    np.set_printoptions(precision=5)

    print("Rewards:")
    print(rewards)
    print("Simple moving average:")
    print(simple_moving_average(rewards, 40))
    # plot_series(simple_moving_average(rewards, 40), "sma.pdf")
    print("Exponential moving average")
    print(exponential_moving_average(rewards, 0.2))
    # plot_series(exponential_moving_average(rewards, 0.2), "ema.pdf")
    print("Average")
    print(np.sum(rewards) / len(rewards))

    if hasattr(agent, "utility"):
        print()
        # print(agent.utility)
        print()

    if hasattr(agent, "observed_state"):
        print("observed_state")
        observed_state = agent.observed_state
        plt.show()
        for k, v in observed_state.items():
            plt.plot(v, label=k)
            plt.legend(labels=["Four, 21: STAND", "Four, 21: HIT", "Ace, 5+Ace: STAND", "Ace, 5+Ace: HIT"])
        plt.savefig("observed_state.pdf")

    # to use this plot function you have to install matplotlib
    # use conda install matplotlib
    # plot_series(simple_moving_average(rewards, 40), "reward_mv_avg.pdf")


# check Wikipedia: https://en.wikipedia.org/wiki/Moving_average
def simple_moving_average(x, N):
    mean = np.zeros(len(x) - N + 1)
    sum = np.sum(x[0:N])
    for i in range(len(mean) - 1):
        mean[i] = sum
        sum -= x[i]
        sum += x[i + N]
    mean[len(mean) - 1] = sum
    return mean / N


# check Wikipedia: https://en.wikipedia.org/wiki/Moving_average
def exponential_moving_average(x, alpha):
    mean = np.zeros(len(x))
    mean[0] = x[0]
    for i in range(1, len(x)):
        mean[i] = alpha * x[i] + (1.0 - alpha) * mean[i - 1]
    return mean


# you can use this function to get a plot
# you need first to install matplotlib (conda install matplotlib)
# and then uncomment this function and lines 1-3
def plot_series(arr, filename=None):
    plt.plot(arr)
    if filename is not None:
        plt.savefig(filename)
        plt.legend()
        plt.show()
