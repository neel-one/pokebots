from poke_env.player.player import Player
import numpy as np
import torch

class ModelPlayer(Player):
    '''
    ModelPlayer is a trained torch nn.Model. Implements the predict function on an embed_battle object.
    '''
    def __init__(self, model, env_player, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model
        self.env_player = env_player

    def choose_move(self, battle):
        with torch.no_grad():
            embed_battle = self.env_player.embed_battle(battle)
            embed_battle = embed_battle.reshape(1,1,embed_battle.shape[0])
            vals = self.model(torch.from_numpy(embed_battle))
            action = np.argmax(vals)
            return self.env_player._action_to_move(action, battle)

class ModelPlayerTF(Player):

    '''
    ModelPlayerTF is a trained tf/keras model. Implements the predict function on an embed_battle object.
    '''
    def __init__(self, model, env_player, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model
        print(model.summary())
        self.env_player = env_player

    def choose_move(self, battle):
        embed_battle = self.env_player.embed_battle(battle)
        embed_battle = embed_battle.reshape(1,1,embed_battle.shape[0])
        vals = self.model.predict(embed_battle)
        action = np.argmax(vals)
        return self.env_player._action_to_move(action, battle)
