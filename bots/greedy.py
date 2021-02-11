import asyncio
from poke_env.player.player import Player
from poke_env.player.random_player import RandomPlayer
from poke_env.environment.pokemon_type import PokemonType
import time
import json
from max_damage import MaxDamagePlayer
#with open('type_chart.json','r') as f:
#    type_chart = json.load(f) 

class Greedy(Player):
    def choose_move(self, battle):
        if battle.available_moves:
            opp = battle.opponent_active_pokemon
            best_move = max(battle.available_moves, key=lambda move: move.base_power*move.type.damage_multiplier(opp.type_1, opp.type_2))
            #print(best_move.type, opp.type_1, opp.type_2)
            #print(best_move.type.damage_multiplier(opp.type_1, opp.type_2))
            return self.create_order(best_move)
        else:
            return self.choose_random_move(battle)

async def main():
    start = time.time()

    #random_player = RandomPlayer(battle_format='gen8randombattle')
    greedy = Greedy(battle_format='gen8randombattle')
    opponent = MaxDamagePlayer(battle_format='gen8randombattle')
    await greedy.battle_against(opponent,n_battles=50)

    print(f'Greedy player won {greedy.n_won_battles}; this process took {time.time()-start} seconds')

if __name__=='__main__':
    #print(type_chart)
    asyncio.get_event_loop().run_until_complete(main())