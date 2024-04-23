"""Implement training loop."""
import numpy as np
import torch
import torch.nn as nn
from torch.nn.modules.linear import Linear
import torch.optim as optim
import torch.nn.functional as F
import json
from threading import Thread
import time

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

class RetryingErrorCount:
    # TODO: make this a context manager, also use a better exception class
    def __init__(self, max_retries, timeout=600):
        self.errors = 0
        self.max_retries = max_retries
        self.timeout = timeout
        self.thread = None
        self.stop = False

    def _listen_func(self, start_time):
        while not self.stop:
            if time.time() - start_time > self.timeout:
                self.errors += 1
                if self.errors >= self.max_retries:
                    raise Exception
                raise TimeoutError
            time.sleep(30)

    def listen(self):
        self.stop = False
        self.thread = Thread(target=self._listen_func, args=(time.time(),))
        self.thread.start()

    def stop(self):
        self.stop = True
        self.thread.join()

def main():
    scores = []

    optimizer = {
        'class': optim.Adam,
        'params': {
            'lr': .00025
        }
    }

# Settings
# history_length = 4

# Use this later, but not now...
    with open('poke/config/config.json') as fp:
        config = json.load(fp)

    train_frequency = 4
    evaluation_frequency = 5000
    target_update_frequency = 500
    initial_replay_size = 5000
    max_replay_size = 50000
    test_samples = 100
    max_steps = 200000
    test_episodes = 50


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
        loss=F.mse_loss,
        dropout=False,
        use_cuda=False
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
    nme = Greedy(battle_format='gen8randombattle',
                    start_timer_on_battle_start=True)

# Training loop
# Fill replay memory with random dataset
    print_epoch(0)
    mdp.start_battles(nme)
# core.learn(n_steps=initial_replay_size,
#           n_steps_per_fit=initial_replay_size)
    core.learn(n_episodes=initial_replay_size,
            n_episodes_per_fit=initial_replay_size)
    mdp.end_battles()
# Evaluate initial policy
    pi.set_epsilon(epsilon_test)
# mdp.set_episode_end(False)
    mdp.start_battles(nme)
    dataset = core.evaluate(n_episodes=test_episodes)
    mdp.end_battles()
    scores.append(get_stats(dataset))

    retry_manager = RetryingErrorCount(max_retries=10, timeout=5*600)
    N_STEPS = max_steps // evaluation_frequency + 1
    for n_epoch in range(1, N_STEPS):
        try:
            retry_manager.listen()
            if n_epoch % 5 == 0 or n_epoch == N_STEPS-1:
                torch.save(core.agent.approximator.model.network, f'checkpoints/torch/checkpt_epoch{n_epoch}')
                core.agent.save(f'checkpoints/mushroom/checkpt_epoch{n_epoch}')
            print_epoch(n_epoch)
            print('- Learning:')
            # learning step
            pi.set_epsilon(epsilon)
            # mdp.set_episode_end(True)
            mdp.start_battles(nme)
            # core.learn(n_steps=evaluation_frequency,
            #            n_steps_per_fit=train_frequency)
            core.learn(n_episodes=evaluation_frequency,
                    n_steps_per_fit=train_frequency)
            mdp.end_battles()

            print('- Evaluation:')
            # evaluation step
            pi.set_epsilon(epsilon_test)
            # mdp.set_episode_end(False)
            mdp.start_battles(nme)
            # dataset = core.evaluate(n_steps=test_samples)
            dataset = core.evaluate(n_episodes=test_episodes)
            mdp.end_battles()
            scores.append(get_stats(dataset))
            retry_manager.stop()
        except Exception as e:
            # Right now the main issue is KeyboardInterrupt from a gnarly deadlock bug
            print(f'Exception {e} happened in training loop, saving state')
            torch.save(core.agent.approximator.model.network, f'checkpoints/torch/checkpt_epoch{n_epoch}')
            core.agent.save(f'checkpoints/mushroom/checkpt_epoch{n_epoch}')

            if not isinstance(e, TimeoutError):
                break

if __name__ == '__main__':
    main()
