import asyncio
from threading import Thread
from mushroom_rl.core import Environment
from mushroom_rl.core.environment import MDPInfo
from mushroom_rl.utils.spaces import Box, Discrete
from poke_env.data import to_id_str
from poke_env.environment.abstract_battle import AbstractBattle
from poke_env.environment.battle import Battle
from poke_env.player.battle_order import BattleOrder
from poke_env.player.env_player import Gen8EnvSinglePlayer
from poke_env.player.player import Player
from poke_env.player_configuration import PlayerConfiguration
from utils import battle_to_state_min, battle_to_state_max, battle_to_state_helper
from typing import Any, Callable, Union, Optional
import numpy as np

nactions = 13


# Default observation and action space defined from observation definition in utils.py
observation_space = Box(battle_to_state_min(), battle_to_state_max())
# For gen8, max of 4 local moves, 4 dynamax moves, 5 switches = 13 actions
# Does this serialization make sense? order matters and things change every time, hmm
action_space = Discrete(nactions)

class ShowdownEnvironment(Environment, Gen8EnvSinglePlayer):
    _ACTION_SPACE = list(range(13)) # override implementation of Gen8EnvSinglePlayer

    def __init__(self, 
        player_configuration: Optional[PlayerConfiguration] = None,
        mdp_info: MDPInfo = None,
        embed_battle_func: Callable = battle_to_state_helper,
        observation_space: Union[Box,Discrete] = observation_space, 
        action_space: Union[Box,Discrete] = action_space, 
        gamma: float = .99, 
        horizon: int = 200,
        start_timer_on_battle_start: bool = True,
        **kwargs
    ):
        if mdp_info is None:
            mdp_info = MDPInfo(observation_space, action_space, gamma, horizon)

        self._embed_battle_func = embed_battle_func

        Environment.__init__(self, mdp_info)

        Gen8EnvSinglePlayer.__init__(self, 
            player_configuration, 
            start_timer_on_battle_start=start_timer_on_battle_start,
            **kwargs
        )

    def _action_to_move(self, action: int, battle: Battle) -> BattleOrder:
        # print(action, battle)
        if isinstance(action, np.ndarray):
            action = action[0]
        # TODO: pass in array of actions by probability to stop from making completely random move
        if action < 4 and action < len(battle.available_moves) and not battle.force_switch:
            return self.create_order(battle.available_moves[action])
        elif battle.can_dynamax and 0 <= action - 4 < len(battle.available_moves) and not battle.force_switch:
            return self.create_order(battle.available_moves[action-4], dynamax=True)
        elif 0 <= action - 8 < len(battle.available_switches):
            return self.create_order(battle.available_switches[action-8])
        else:
            return self.choose_random_move(battle)

    def reset(self, state=None):
        return Gen8EnvSinglePlayer.reset(self)

    def step(self, action):
        return Gen8EnvSinglePlayer.step(self, action)
    
    def embed_battle(self, battle: AbstractBattle) -> Any:
        return self._embed_battle_func(battle)

    def start_battles(self, opponent):
        
        # Similar to self.play_against, but the battles on on a different thread
        # While end_battles signals battles to end
        self._start_new_battle = True
        loop = asyncio.get_event_loop()

        async def launch_battles(opponent: Player):
            battles_coroutine = asyncio.gather(
                self.send_challenges(
                    opponent=to_id_str(opponent.username),
                    n_challenges=1,
                    to_wait=opponent.logged_in,
                ),
                opponent.accept_challenges(
                    opponent=to_id_str(self.username), n_challenges=1
                ),
            )
            await battles_coroutine

        def loop_battles():
            # try:
            #     loop = asyncio.get_event_loop()
            # except RuntimeError:
            # loop = asyncio.new_event_loop()
            # asyncio.set_event_loop(loop)
            while self._start_new_battle:
                loop.run_until_complete(launch_battles(opponent))
                
        battle_thread = Thread(
            target = loop_battles
        )

        battle_thread.start()

        # return battle_thread

    def end_battles(self):
        self._start_new_battle = False
        while True:
            try:
                self.complete_current_battle()
                self.reset()
            except OSError:
                break
        # battle_thread.join(), don't need battle thread, complete_current_battle finishes thread