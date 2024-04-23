import time
from poke import Greedy, ModelPlayer, ShowdownEnvironment
import asyncio
import torch
from pathlib import Path

async def main():
    start = time.time()

    #random_player = RandomPlayer(battle_format='gen8randombattle')
    #model = torch.load('checkpoints/checkpt_epoch5')
    n_battles = 50
    models = Path('checkpoints/torch').glob('*')
    model = torch.load(str(max(models)))
    greedy = Greedy(battle_format='gen8randombattle', start_timer_on_battle_start=True)
    opponent = ModelPlayer(model=model, env_player=ShowdownEnvironment(), battle_format='gen8randombattle')
    await greedy.battle_against(opponent,n_battles=n_battles)

    print(f'Greedy player won {greedy.n_won_battles}; this process took {time.time()-start} seconds')
    print(f'ModelPlayer won {opponent.n_won_battles} for a win rate of {opponent.n_won_battles/n_battles}')

if __name__=='__main__':
    #print(type_chart)
    asyncio.get_event_loop().run_until_complete(main())
