from mushroom_rl.core import Environment
from mushroom_rl.core.environment import MDPInfo
from mushroom_rl.utils.spaces import Box, Discrete
from poke_env.environment.battle import Battle
from poke_env.player.battle_order import BattleOrder
from poke_env.player.env_player import Gen8EnvSinglePlayer
from poke_env.player_configuration import PlayerConfiguration
from utils import battle_to_state_min, battle_to_state_max
from typing import Union, Optional

nactions = 13


# Default observation and action space defined from observation definition in utils.py
observation_space = Box(battle_to_state_min, battle_to_state_max)
# For gen8, max of 4 local moves, 4 dynamax moves, 5 switches = 13 actions
# Does this serialization make sense? order matters and things change every time, hmm
action_space = Discrete(nactions)

class ShowdownEnvironment(Environment, Gen8EnvSinglePlayer):
    _ACTION_SPACE = list(range(13)) # override implementation of Gen8EnvSinglePlayer

    def __init__(self, 
        player_configuration: Optional[PlayerConfiguration] = None,
        mdp_info: MDPInfo = None,
        observation_space: Union[Box,Discrete] = observation_space, 
        action_space: Union[Box,Discrete] = action_space, 
        gamma: float = .99, 
        horizon: int = 200,
        start_time_on_battle_start: bool = True,
        **kwargs
    ):
        if mdp_info is None:
            mdp_info = MDPInfo(observation_space, action_space, gamma, horizon)

        super().__init__(mdp_info, 
            player_configuration, 
            start_time_on_battle_start=start_time_on_battle_start,
            **kwargs
        )

    def _action_to_move(self, action: int, battle: Battle) -> BattleOrder:
        # TODO: pass in array of actions by probability to stop from making completely random move
        if action < 4 and action < len(battle.available_moves) and not battle.force_switch:
            return self.create_order(battle.available_moves[action])
        elif battle.can_dynamax and 0 <= action - 4 < len(battle.available_moves) and not battle.force_switch:
            return self.create_order(battle.available_moves[action-4], dynamax=True)
        elif action - 8 < len(battle.available_switches):
            return self.create_order(battle.available_switches[action-8])
        else:
            return self.choose_random_move(battle)

    def reset(self, state=None):
        return Gen8EnvSinglePlayer.reset(self)

    def step(self, action):
        return Gen8EnvSinglePlayer.step(self, action)

