import time
from poke import Greedy, ModelPlayer, ShowdownEnvironment
import asyncio
import torch

async def main():
    start = time.time()

    #random_player = RandomPlayer(battle_format='gen8randombattle')
    model = torch.load('checkpoints/checkpt_epoch5')
    
    greedy = Greedy(battle_format='gen8randombattle', start_timer_on_battle_start=True)
    opponent = ModelPlayer(model=model, env_player=ShowdownEnvironment(), battle_format='gen8randombattle')
    await greedy.battle_against(opponent,n_battles=10)

    print(f'Greedy player won {greedy.n_won_battles}; this process took {time.time()-start} seconds')

if __name__=='__main__':
    #print(type_chart)
    asyncio.get_event_loop().run_until_complete(main())