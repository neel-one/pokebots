from poke_env.player_configuration import PlayerConfiguration
from poke_env.server_configuration import ShowdownServerConfiguration
from greedy import Greedy
import asyncio
from bs4 import BeautifulSoup
import requests
import matplotlib.pyplot as plt
import threading
import time
from collections import OrderedDict

finished = False

def getElo(username, format='gen8randombattle'):
    username = username.replace(' ','')
    page = requests.get(f'https://pokemonshowdown.com/users/{username}')
    soup = BeautifulSoup(page.content, 'html.parser')
    elt = soup.find(string=format).parent
    rating = elt.next_sibling.contents[0].string
    return int(rating)

def plotElo(username, num_games=5, file=None, format='gen8randombattle'):
    global finished
    elo = []
    while not finished:
        try:
            elo.append(getElo(username, format))
            print(f'ELO: {elo[-1]}')
        except:
            pass
        finally:
            time.sleep(30)
    print("Finished battles, generating PLOT")
    elo = [elo[i] for i in range(len(elo)) if (i==0) or elo[i] != elo[i-1]]
    plt.plot([i for i in range(1,num_games+1)], elo)
    plt.ylabel('ELO')
    plt.xlabel('Game')
    if file is not None:
        plt.savefig(file)
    else:
        plt.show()

async def main():
    player = Greedy(player_configuration=PlayerConfiguration('a very greedy bot', 'password'),
                    server_configuration=ShowdownServerConfiguration,
                    start_timer_on_battle_start=True
                    )
    #await player.accept_challenges(None, 10)
    t = threading.Thread(target=plotElo, args=('a very greedy bot',), kwargs={'file':'GreedyElo.png'})
    t.start()
    await player.ladder(5)
    time.sleep(120)
    global finished
    finished = True
    print("Finished Battles")

    #await ladderAndPlotELO(player, 'a very greedy bot', file='GreedyElo.png')

if __name__=='__main__':
    asyncio.get_event_loop().run_until_complete(main())