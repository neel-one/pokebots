from poke_env.player import TrainablePlayer
from poke_env.player_configuration import PlayerConfiguration
from poke_env.server_configuration import ServerConfiguration
from poke_env.environment.abstract_battle import AbstractBattle
from typing import Optional, Dict
import numpy as np



class Agent(TrainablePlayer):

    def __init__(
            self,
            player_configuration: Optional[PlayerConfiguration] = None,
            *,
            battle_format: str = 'gen8randombattle',
            max_concurrent_battles: int = 1,
            model = None,
            server_configuration: Optional[ServerConfiguration] = None,
            **kwargs):
        '''
        For other keyword arguments, see PokeEnv docs for TrainablePlayer, this class keeps the important ones for PokeRL
        '''
        super().__init__(player_configuration, battle_format, max_concurrent_battles, model, server_configuration, **kwargs)

    #TODO: define any procedures that can be done in a base class, but not effectively achieved in TrainablePlayer

#Once possible way to represent state from a battle
def battle_to_state_helper(battle: AbstractBattle):
    pkmn = battle.active_pokemon
    opp_pkmn

class AI(Agent):

    @staticmethod
    def init_model():
        pass

    def action_to_move(self, action, battle: AbstractBattle):
        '''
        Translate action to move_order
        '''
        pass
    
    
    def battle_to_state(self, battle: AbstractBattle):
        '''
        Serialize battle object to state vector
        '''
        pass

    #Consider Union[np.array, torch.tensor, etc.]
    # Resulted by network
    def state_to_action(self, state: np.array, battle: AbstractBattle):
        '''
        Output chosen action, given state.
        E.G. In DQN with Epsilon Greedy policy of .2, we output a network result with probability .8, 
        else a random action with probability .2
        '''
        pass

    def replay(self, battle_history: Dict):
        pass
