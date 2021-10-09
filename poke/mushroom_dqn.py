import numpy as np
import torch
from torch._C import _infer_size
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

from mushroom_utils import ShowdownEnvironment
from greedy import Greedy

class Network(nn.Module):
    in_size = 110
    out_size = 13

    def __init__(self, input_shape, output_shape, **kwargs):
        super().__init__()

        assert (self.in_size,) == input_shape
        assert (self.out_size,) == output_shape

        self.h1 = nn.Linear(self.in_size, 150)
        self.h2 = nn.Linear(150, 70)
        self.h3 = nn.Linear(70, 30)
        self.h4 = nn.Linear(30, self.out_size)
        
        nn.init.xavier_uniform_(self.h1.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.h2.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.h3.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.h4.weight, gain=nn.init.calculate_gain('linear'))

    def forward(self, state, action=None):
        x = F.relu(self.h1(state))
        x = F.relu(self.h2(x))
        x = F.relu(self.h3(x))
        qs = self.h4(x)
        if action is None:
            return qs
        else:
            q_acted = torch.squeeze(qs.gather(1, action.long()))
            return q_acted

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
#history_length = 4
train_frequency = 4
evaluation_frequency = 250
target_update_frequency = 100
initial_replay_size = 150
max_replay_size = 50000
test_samples = 100
max_steps = 10000


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
greedy = Greedy(battle_format='gen8randombattle', start_timer_on_battle_start=True)
# 
# Training loop
# Fill replay memory with random dataset
print_epoch(0)
mdp.start_battles(greedy)
core.learn(n_steps=initial_replay_size,
           n_steps_per_fit=initial_replay_size)
mdp.end_battles()
# Evaluate initial policy
pi.set_epsilon(epsilon_test)
#mdp.set_episode_end(False)
mdp.start_battles(greedy)
dataset = core.evaluate(n_steps=test_samples)
mdp.end_battles()
scores.append(get_stats(dataset))

for n_epoch in range(1, max_steps // evaluation_frequency + 1):
    print_epoch(n_epoch)
    print('- Learning:')
    # learning step
    pi.set_epsilon(epsilon)
    #mdp.set_episode_end(True)
    mdp.start_battles(greedy)
    core.learn(n_steps=evaluation_frequency,
               n_steps_per_fit=train_frequency)
    mdp.end_battles()

    print('- Evaluation:')
    # evaluation step
    pi.set_epsilon(epsilon_test)
    #mdp.set_episode_end(False)
    mdp.start_battles(greedy)
    dataset = core.evaluate(n_steps=test_samples)
    mdp.end_battles()
    scores.append(get_stats(dataset))