from poke_env.player_configuration import PlayerConfiguration
from poke_env.server_configuration import ShowdownServerConfiguration
from greedy import Greedy
import asyncio

async def main():
    player = Greedy(player_configuration=PlayerConfiguration('a very greedy bot', 'password'),
                    server_configuration=ShowdownServerConfiguration
                    )
    await player.accept_challenges(None, 10)

if __name__=='__main__':
    asyncio.get_event_loop().run_until_complete(main())