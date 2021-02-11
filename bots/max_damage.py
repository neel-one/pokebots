import asyncio
#import sys, os
#sys.path.insert(0, os.path.join(os.path.abspath(os.path.join(os.getcwd(),os.pardir)), 'poke_env'))
#print(sys.path)
from poke_env.player.player import Player
from poke_env.player.random_player import RandomPlayer
import time

class MaxDamagePlayer(Player):
    def choose_move(self, battle):
        if battle.available_moves:
            best_move = max(battle.available_moves, key=lambda move: move.base_power)
            return self.create_order(best_move)
        else:
            return self.choose_random_move(battle)

async def main():
    start = time.time()

    #random_player = RandomPlayer(battle_format='gen8randombattle')
    max_damage_player = MaxDamagePlayer(battle_format='gen8randombattle')
    opponent = MaxDamagePlayer(battle_format='gen8randombattle')
    await max_damage_player.battle_against(opponent,n_battles=5)

    print(f'Max damage player won {max_damage_player.n_won_battles}; this process took {time.time()-start} seconds')

if __name__=='__main__':
    asyncio.get_event_loop().run_until_complete(main())
