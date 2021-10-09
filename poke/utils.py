from poke_env.environment.move import Move
from poke_env.environment.pokemon import Pokemon
from poke_env.environment.abstract_battle import AbstractBattle
from typing import Optional
import numpy as np

move_to_vec_min = np.zeros(7)
move_to_vec_max = np.array([1,250,6,5,6,1,7])
def move_to_vec_helper(move: Move, pkmn: Optional[Pokemon] = None):
  acc = move.accuracy #float
  base_power = move.base_power #int
  #boosts = move.boosts # dict
  crit_ratio = move.crit_ratio
  expected_hits = move.expected_hits
  priority = move.priority #int
  terrain = 0 if not move.terrain else 1 #optional str
  weather = move.weather.value if move.weather else 0
  return np.array(list(locals().values())[2:])

def pkmn_to_vec_min():
  s = np.zeros(6)
  level = 1 #int
  #stats = pkmn.stats #dict
  status = 0 #int
  status_counter = 0 #int
  hp = 0 #float, perhaps better than current_hp which differs depending on your pkmn or opponent pkmn
  # effects = pkmn.effects #dict --> probably ignore, fairly hard to serialize...
  alive = 0 #int
  first_turn = 0 #int
  #item = pkmn.item #str, probably can be learned from context
  #must_recharge = pkmn.must_recharge #bool, probably dont need, nobody uses recharge moves anyways...
  protect_counter = 0 #int

  #type info
  type_1 = 0 #int
  type_2 = 0 #int

  s = np.append(s, [level, status, status_counter, hp, alive, first_turn, protect_counter, type_1, type_2])
  #move info
  mvs = np.zeros((4,7))
  for i in range(4):
    move_arr = move_to_vec_min
    mvs[i] = move_arr
  return np.append(s, mvs.flatten()) 

def pkmn_to_vec_max():
  s = np.full(6, 800) #np.array
  level = 100 #int
  #stats = pkmn.stats #dict
  status = 7 
  status_counter = 15
  hp = 1 #float, perhaps better than current_hp which differs depending on your pkmn or opponent pkmn
  # effects = pkmn.effects #dict --> probably ignore, fairly hard to serialize...
  alive = 1
  first_turn = 1
  #item = pkmn.item #str, probably can be learned from context
  #must_recharge = pkmn.must_recharge #bool, probably dont need, nobody uses recharge moves anyways...
  protect_counter = 5

  #type info
  type_1 = 18
  type_2 = 18

  s = np.append(s, [level, status, status_counter, hp, alive, first_turn, protect_counter, type_1, type_2])
  #move info
  mvs = np.zeros((4,7))
  for i in range(4):
    move_arr = move_to_vec_max
    mvs[i] = move_arr

  return np.append(s, mvs.flatten())

def pkmn_to_vec_helper(pkmn: Pokemon):
  #meta info
  ### I suppose, some of the information such as ability/etc can be learned from base stats/types
  #ability = pkmn.ability #str, somehow find a way to turn this into an int; we can ignore this for now...
  #active = pkmn.active #bool

  multiplier = {-6: 2/8, -5: 2/7, -4: 2/6, -3: 2/5, -2: 2/4, -1: 2/3, 0: 1, 1: 3/2, 2: 4/2, 3: 5/2, 4: 3, 5: 7/2, 6: 4}
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
    if i >= 4:
      break
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

  return np.concatenate((pkmn_to_vec_helper(pkmn), pkmn_to_vec_helper(opp_pkmn),
                        f, side_conds, opp_side_conds, [weather, can_dyna, dyna_turns_left,
                        opp_can_dyna, opp_dyna_turns_left]), dtype=np.float32)

def battle_to_state_min():
  #field, weather, and conditions
  f = np.zeros(5) # np.array

  side_conds = np.zeros(7) #np.array for the sake of simplicity we only care about screens and hazards
  opp_side_conds = np.zeros(7) #np.array

  weather = 0

  #dynamax details
  can_dyna = 0
  dyna_turns_left = 0 #int
  opp_can_dyna = 0 #int
  opp_dyna_turns_left = 0 #int

  return np.concatenate((pkmn_to_vec_min(), pkmn_to_vec_min(),
                        f, side_conds, opp_side_conds, [weather, can_dyna, dyna_turns_left,
                        opp_can_dyna, opp_dyna_turns_left]))

def battle_to_state_max():
  #field, weather, and conditions
  f = np.full(5, 8) # np.array

  side_conds = np.full(7,3) #np.array for the sake of simplicity we only care about screens and hazards
  opp_side_conds = np.full(7,3) #np.array

  weather = 7

  #dynamax details
  can_dyna = 1
  dyna_turns_left = 3
  opp_can_dyna = 1
  opp_dyna_turns_left = 3

  return np.concatenate((pkmn_to_vec_max(), pkmn_to_vec_max(),
                        f, side_conds, opp_side_conds, [weather, can_dyna, dyna_turns_left,
                        opp_can_dyna, opp_dyna_turns_left]))
