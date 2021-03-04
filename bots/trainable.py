from poke_env.player import TrainablePlayer




class Agent(TrainablePlayer):

    def __init__(
            self,
            player_configuration: Optional[PlayerConfiguration] = None,
            *,
            battle_format: str = 'gen8randombattle',
            max_concurrent_battles: int = 1,
            model = None,
            server_configuration: Optional[ServerConfiguration] = None
            **kwargs):
        '''
        For other keyword arguments, see PokeEnv docs for TrainablePlayer, this class keeps the important ones for PokeRL
        '''
        super().__init__(player_configuration, battle_format, max_concurrent_battles, model, server_configuration, **kwargs)

    #TODO: define any procedures that can be done in a base class, but not effectively achieved in TrainablePlayer


class AI(Agent):

    @staticmethod
    def init_model():
        pass

    def action_to_move(self, action, battle: AbstractBattle):
        pass

    def battle_to_state(self, battle: AbstractBattle):
        pass

    #Consider Union[np.array, torch.tensor, etc.]
    def state_to_action(self, state: np.array, battle: AbstractBattle):
        pass

    def replay(self, battle_history: Dict):
        pass
