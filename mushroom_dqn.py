"""Implement training loop."""
import numpy as np
import torch
import torch.nn as nn
from torch.nn.modules.linear import Linear
import torch.optim as optim
import torch.nn.functional as F

from mushroom_rl.algorithms.value import DQN
from mushroom_rl.approximators.parametric import TorchApproximator
from mushroom_rl.core import Core
from mushroom_rl.environments import Atari
from mushroom_rl.policy import EpsGreedy
from mushroom_rl.utils.dataset import compute_metrics
from mushroom_rl.utils.parameters import LinearParameter, Parameter

from poke.mushroom_utils import ShowdownEnvironment
from poke.greedy import Greedy
from poke.networks import Network


def print_epoch(epoch):
    print('################################################################')
    print('Epoch: ', epoch)
    print('----------------------------------------------------------------')


def get_stats(dataset):
    score = compute_metrics(dataset)
    print(('min_reward: %f, max_reward: %f, mean_reward: %f,'
          ' games_completed: %d' % score))

    return score


scores = []

optimizer = {
    'class': optim.Adam,
    'params': {
        'lr': .00025
    }
}

# Settings
# history_length = 4
train_frequency = 120
evaluation_frequency = 5000
target_update_frequency = 500
initial_replay_size = 150
max_replay_size = 50000
test_samples = 100
max_steps = 10000
test_episodes = 10


# PO-MDP
mdp = ShowdownEnvironment()

# Policy
epsilon = LinearParameter(value=1.,
                          threshold_value=.1,
                          n=1000000)
epsilon_test = Parameter(value=.05)
epsilon_random = Parameter(value=1)
pi = EpsGreedy(epsilon=epsilon_random)

# Approximator
approximator_params = dict(
    network=Network,
    input_shape=(110,),
    output_shape=(13,),
    n_actions=13,
    optimizer=optimizer,
    loss=F.mse_loss
)

approximator = TorchApproximator

# Agent
algorithm_params = dict(
    batch_size=32,
    target_update_frequency=target_update_frequency,
    replay_memory=None,
    initial_replay_size=initial_replay_size,
    max_replay_size=max_replay_size
)

agent = DQN(mdp.info, pi, approximator,
            approximator_params=approximator_params,
            **algorithm_params)

# Algorithm
core = Core(agent, mdp)

# Opponent
greedy = Greedy(battle_format='gen8randombattle',
                start_timer_on_battle_start=True)

# Training loop
# Fill replay memory with random dataset
print_epoch(0)
mdp.start_battles(greedy)
core.learn(n_steps=initial_replay_size,
           n_steps_per_fit=initial_replay_size)
mdp.end_battles()
# Evaluate initial policy
pi.set_epsilon(epsilon_test)
# mdp.set_episode_end(False)
mdp.start_battles(greedy)
dataset = core.evaluate(n_steps=test_samples)
mdp.end_battles()
scores.append(get_stats(dataset))

for n_epoch in range(1, max_steps // evaluation_frequency + 1):
    print_epoch(n_epoch)
    print('- Learning:')
    # learning step
    pi.set_epsilon(epsilon)
    # mdp.set_episode_end(True)
    mdp.start_battles(greedy)
    core.learn(n_steps=evaluation_frequency,
               n_steps_per_fit=train_frequency)
    mdp.end_battles()

    print('- Evaluation:')
    # evaluation step
    pi.set_epsilon(epsilon_test)
    # mdp.set_episode_end(False)
    mdp.start_battles(greedy)
    # dataset = core.evaluate(n_steps=test_samples)
    dataset = core.evaluate(n_episodes=test_episodes)
    mdp.end_battles()
    scores.append(get_stats(dataset))
