import asyncio
from poke_env.player.player import Player
from poke_env.player.random_player import RandomPlayer
from poke_env.environment.pokemon_type import PokemonType
from poke_env.environment.pokemon import Pokemon
from poke_env.environment.move import Move
from poke_env.environment.abstract_battle import AbstractBattle
import numpy as np
import time
import json
from functools import cmp_to_key
import sys
from typing import Optional
import logging
import poke.utils
#from max_damage import MaxDamagePlayer
#with open('type_chart.json','r') as f:
#    type_chart = json.load(f)
class Greedy(Player):
    def choose_move(self, battle):
        # logging.warn(f'Available moves: {battle.available_moves}')
        opp = battle.opponent_active_pokemon
        def calcDmg(move):
            dmg = move.base_power*move.type.damage_multiplier(opp.type_1, opp.type_2)
            if battle.active_pokemon.type_1 == move.type or battle.active_pokemon.type_2 == move.type:
                return 1.5 * dmg
            return dmg
        def cmpMoves(move1, move2):
            return calcDmg(move1) - calcDmg(move2)
        if battle.available_moves:
            best_move = max(battle.available_moves, key=cmp_to_key(cmpMoves))
            return self.create_order(best_move)
        else:
            return self.choose_random_move(battle)

def move_to_vec_helper(move: Move, pkmn: Optional[Pokemon] = None):
  acc = move.accuracy #float
  base_power = move.base_power #int
  #boosts = move.boosts # dict
  crit_ratio = move.crit_ratio
  expected_hits = move.expected_hits
  priority = move.priority #int
  terrain = 0 if not move.terrain else 1 #optional str
  weather = move.weather.value if move.weather else 0
  # python hack because i am lazy
  return np.array(list(locals().values())[2:])

multiplier = {-6: 2/8, -5: 2/7, -4: 2/6, -3: 2/5, -2: 2/4, -1: 2/3, 0: 1, 1: 3/2, 2: 4/2, 3: 5/2, 4: 3, 5: 7/2, 6: 4}

def pkmn_to_vec_helper(pkmn: Pokemon):
  #meta info
  ### I suppose, some of the information such as ability/etc can be learned from base stats/types
  #ability = pkmn.ability #str, somehow find a way to turn this into an int; we can ignore this for now...
  #active = pkmn.active #bool

  s = np.zeros(6) #np.array
  base_stats = pkmn.base_stats #dict, base stats and level may be better than stats because of opponent information...
  boosts = pkmn.boosts #dict
  for i, stat in enumerate(["hp","atk","def","spa","spd","spe"]):
    s[i] = base_stats[stat]
    if stat in boosts:
      #2/8 2/7	2/6	2/5	2/4	2/3	2/2	3/2	4/2	5/2	6/2	7/2	8/2
      s[i] *= multiplier[boosts[stat]]

  level = pkmn.level #int
  #stats = pkmn.stats #dict
  status = pkmn.status.value if pkmn.status is not None else 0 #int
  status_counter = pkmn.status_counter #int
  hp = pkmn.current_hp_fraction #float, perhaps better than current_hp which differs depending on your pkmn or opponent pkmn
  # effects = pkmn.effects #dict --> probably ignore, fairly hard to serialize...
  alive = 0 if pkmn.fainted else 1 #int
  first_turn = 1 if pkmn.first_turn else 0 #int
  #item = pkmn.item #str, probably can be learned from context
  #must_recharge = pkmn.must_recharge #bool, probably dont need, nobody uses recharge moves anyways...
  protect_counter = pkmn.protect_counter #int

  #type info
  type_1 = pkmn.type_1.value #int
  type_2 = pkmn.type_2.value if pkmn.type_2 else 0 #int

  s = np.append(s, [level, status, status_counter, hp, alive, first_turn, protect_counter, type_1, type_2])
  #move info
  moves = pkmn.moves #dict
  mvs = np.zeros((4,7))
  for i, move in enumerate(moves.values()):
    move_arr = move_to_vec_helper(move)
    mvs[i] = move_arr

  return np.append(s, mvs.flatten())

def side_conds_helper(side_conds):
  c = np.zeros(7)
  for i, cond in enumerate([1,9,12,14,15,16,18]):
    c[i] = side_conds[cond] if cond in side_conds else 0
  return c

def battle_to_state_helper(battle: AbstractBattle):
  #pkmn info
  ### Perhaps ignore team in state calculations, and use algorithm that has context... say LSTM???
  pkmn = battle.active_pokemon #pkmn
  #team = battle.team #dict
  opp_pkmn = battle.opponent_active_pokemon #pkmn
  #opp_team = battle.opponent_team #dict

  #field, weather, and conditions
  fields = battle.fields #dict, only care about terrain and trick room
  f = np.zeros(5) # np.array
  for i, field in enumerate([1, 2, 6, 9, 10]):
      f[i] = fields[field] - battle.turn if field in fields else 0

  side_conds = side_conds_helper(battle.side_conditions) #np.array for the sake of simplicity we only care about screens and hazards
  opp_side_conds = side_conds_helper(battle.opponent_side_conditions) #np.array

  weather = next(iter(battle.weather)).value if battle.weather is not None and battle.weather != {} else 0 #int

  #dynamax details
  can_dyna = 1 if battle.can_dynamax else 0 # int
  dyna_turns_left = battle.dynamax_turns_left if battle.dynamax_turns_left is not None else 0 #int
  opp_can_dyna = 1 if battle.opponent_can_dynamax else 0 #int
  opp_dyna_turns_left = battle.opponent_dynamax_turns_left if battle.opponent_dynamax_turns_left is not None else 0 #int

  return torch.from_numpy(np.concatenate((pkmn_to_vec_helper(pkmn), pkmn_to_vec_helper(opp_pkmn),
                        f, side_conds, opp_side_conds, [weather, can_dyna, dyna_turns_left,
                        opp_can_dyna, opp_dyna_turns_left])))

  #return np.concatenate((pkmn_to_vec_helper(pkmn), []))
l = np.full((110,),np.inf)
u = np.full((110,),-np.inf)
class Test(Player):
    ignore_prop = set(('MESSAGES_TO_IGNORE', 'battle_tag', 'can_mega_evolve', 'can_z_move', 'players', 'rqid',
                    'logger','move_on_next_request','player_role','rating', 'team_size', 'team_preview'))
    def choose_move(self, battle):
        global l,u
        # for f in dir(battle):
        #     if f[0] != '_' and f not in self.ignore_prop:
        #         print(f, end=': ')
        #         print(getattr(battle, f))
        start = time.time()
        state = utils.battle_to_state_helper(battle)
        end = time.time()

        l = np.minimum(l,state)
        u = np.maximum(u,state)
        print(len(state), f'time: {end-start}')
        #print(state.shape)

        return self.choose_random_move(battle)

async def main():
    start = time.time()

    #random_player = RandomPlayer(battle_format='gen8randombattle')
    greedy = Greedy(battle_format='gen8randombattle', start_timer_on_battle_start=True)
    opponent = Test(battle_format='gen8randombattle')
    await greedy.battle_against(opponent,n_battles=1000)

    print(f'Greedy player won {greedy.n_won_battles}; this process took {time.time()-start} seconds')

    print(list(l))
    print(list(u))

if __name__=='__main__':
    #print(type_chart)
    asyncio.get_event_loop().run_until_complete(main())
