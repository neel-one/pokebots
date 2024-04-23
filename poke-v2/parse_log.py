import re
import requests
from pprint import pprint  # For better visualization of the dataset
from copy import deepcopy
import json
import sys

# TODO: the dataset for each player should prefetch their entire pokemon 
# state in the first pre_action_state
# The opponent doesn't need to know that info (they're known info is just they're own previous post action state)
# To do this we'll probably need to do some post processing of the result list
# Precisely: for all players p, for all turns with player_id = p, p's pre_action_state and post_action_state contains all of their pokemon

# TODO: how to encode available moves in the dataset?
# one option is that we don't, but at inference time we pick the top probability move
# that the pokemon can do

# Function to update the game state based on an action
def update_game_state(state, action):
    if action['type'] == 'damage' or action['type'] == 'heal':
        player, pokemon = action['target'].split(': ')
        if 'fnt' in action['new_hp']:
            hp_current = hp_max = action['new_hp']
        else:
            hp_current, hp_max = action['new_hp'].split('/')
        state[player][pokemon] = {'hp_current': hp_current, 'hp_max': hp_max}
    if action['type'] == 'switch':
        player, pokemon = action['details'][0].split(': ')
        hp_current, hp_max = action['details'][2].split('/')
        state[player][pokemon] = {'hp_current': hp_current, 'hp_max': hp_max}

    # Add more conditions to handle other types of actions and state changes

# Function to parse each turn into a detailed structured format
def parse_detailed_turn(turn_str, last_turn_state):
    lines = turn_str.strip().split('\n')
    current_turn = int(lines[0])
    #game_state = {"p1a": {}, "p2a": {}}
    #game_state = {**last_turn_state}
    game_state = deepcopy(last_turn_state)
    structured_turns = []
    for line in lines[1:]:
        line = line[1:]
        if line.startswith('move') or line.startswith('switch'):
            print(line)
            action_type, details = line.split('|')[0], line.split('|')[1:]
            player_id = 'p1a' if 'p1' in details[0] else 'p2a'

            # for some reason not deep copying is better?
            pre_action_state = last_turn_state  # Copy current state before the action

            # Assuming actions are either 'move' or 'switch', extend as needed
            action = {
                'type': action_type,
                'details': details,
            }

            # Update game state based on action effects here
            update_game_state(game_state, action)

            post_action_state = {**game_state}  # Copy state after the action


            structured_turns.append({
                "turn": current_turn,
                "player_id": player_id,
                "action": action_type,
                "details": details[1] if action_type == 'move' else details[1][:details[1].find(',')],
                "pre_action_state": pre_action_state,
                "post_action_state": post_action_state
            })
        
        elif line.startswith('-damage') or line.startswith('-heal'):
            # Extract target and new hp status
            parts = line.split('|')
            target = parts[1]
            new_hp = parts[2]
            action = {
                'type': 'damage' if 'damage' in line else 'heal',
                'target': target,
                'new_hp': new_hp,
            }
            print(game_state)
            # No need to append to structured_turns since these are state updates, not actions
            update_game_state(game_state, action)
    return structured_turns


game_log = """
|j|☆actualboy
|j|☆gaidepcobird
|t:|1711853420
|gametype|singles
|player|p1|actualboy|266|1152
|player|p2|gaidepcobird|twins-gen4dp|1168
|teamsize|p1|6
|teamsize|p2|6
|gen|9
|tier|[Gen 9] Random Battle
|rated|
|rule|Species Clause: Limit one of each Pokémon
|rule|HP Percentage Mod: HP is shown in percentages
|rule|Sleep Clause Mod: Limit one foe put to sleep
|rule|Illusion Level Mod: Illusion disguises the Pokémon's true level
|
|t:|1711853420
|start
|switch|p1a: Salazzle|Salazzle, L82, F|246/246
|switch|p2a: Mabosstiff|Mabosstiff, L85, F|275/275
|turn|1
|inactive|Battle timer is ON: inactive players will automatically lose when time's up. (requested by gaidepcobird)
|
|t:|1711853443
|move|p1a: Salazzle|Protect|p1a: Salazzle
|-singleturn|p1a: Salazzle|Protect
|move|p2a: Mabosstiff|Psychic Fangs|p1a: Salazzle
|-activate|p1a: Salazzle|move: Protect
|
|upkeep
|turn|2
|
|t:|1711853466
|switch|p1a: Snorlax|Snorlax, L82, F|397/397
|move|p2a: Mabosstiff|Psychic Fangs|p1a: Snorlax
|-damage|p1a: Snorlax|194/397
|
|-heal|p1a: Snorlax|218/397|[from] item: Leftovers
|upkeep
|turn|3
|
|t:|1711853480
|move|p2a: Mabosstiff|Psychic Fangs|p1a: Snorlax
|-damage|p1a: Snorlax|119/397
|move|p1a: Snorlax|Earthquake|p2a: Mabosstiff
|-damage|p2a: Mabosstiff|197/275
|
|-heal|p1a: Snorlax|143/397|[from] item: Leftovers
|upkeep
|turn|4
|
|t:|1711853502
|switch|p2a: Golurk|Golurk, L87|297/297
|move|p1a: Snorlax|Curse|p1a: Snorlax
|-unboost|p1a: Snorlax|spe|1
|-boost|p1a: Snorlax|atk|1
|-boost|p1a: Snorlax|def|1
|
|-heal|p1a: Snorlax|167/397|[from] item: Leftovers
|upkeep
|turn|5
|
|t:|1711853512
|move|p2a: Golurk|Dynamic Punch|p1a: Snorlax
|-supereffective|p1a: Snorlax
|-damage|p1a: Snorlax|0 fnt
|faint|p1a: Snorlax
|
|upkeep
|inactive|actualboy has 120 seconds left.
|
|t:|1711853532
|switch|p1a: Bombirdier|Bombirdier, L85, F|258/258
|turn|6
|
|t:|1711853537
|move|p1a: Bombirdier|Knock Off|p2a: Golurk
|-supereffective|p2a: Golurk
|-damage|p2a: Golurk|0 fnt
|-enditem|p2a: Golurk|Choice Band|[from] move: Knock Off|[of] p1a: Bombirdier
|faint|p2a: Golurk
|
|upkeep
|
|t:|1711853545
|switch|p2a: Mabosstiff|Mabosstiff, L85, F|197/275
|turn|7
|
|t:|1711853560
|switch|p1a: Salazzle|Salazzle, L82, F|246/246
|move|p2a: Mabosstiff|Wild Charge|p1a: Salazzle
|-damage|p1a: Salazzle|43/246
|-damage|p2a: Mabosstiff|146/275|[from] Recoil
|
|-heal|p1a: Salazzle|58/246|[from] item: Leftovers
|upkeep
|turn|8
|
|t:|1711853570
|move|p2a: Mabosstiff|Wild Charge|p1a: Salazzle
|-damage|p1a: Salazzle|0 fnt
|faint|p1a: Salazzle
|-damage|p2a: Mabosstiff|131/275|[from] Recoil
|
|upkeep
|
|t:|1711853587
|switch|p1a: Dachsbun|Dachsbun, L91, M|252/252
|turn|9
|
|t:|1711853609
|move|p2a: Mabosstiff|Wild Charge|p1a: Dachsbun
|-damage|p1a: Dachsbun|190/252
|-damage|p2a: Mabosstiff|115/275|[from] Recoil
|move|p1a: Dachsbun|Body Press|p2a: Mabosstiff
|-supereffective|p2a: Mabosstiff
|-damage|p2a: Mabosstiff|0 fnt
|faint|p2a: Mabosstiff
|
|-heal|p1a: Dachsbun|205/252|[from] item: Leftovers
|upkeep
|inactive|gaidepcobird has 120 seconds left.
|
|t:|1711853632
|switch|p2a: Barraskewda|Barraskewda, L81, F|231/231
|turn|10
|
|t:|1711853642
|move|p2a: Barraskewda|Poison Jab|p1a: Dachsbun
|-supereffective|p1a: Dachsbun
|-damage|p1a: Dachsbun|53/252
|move|p1a: Dachsbun|Play Rough|p2a: Barraskewda
|-damage|p2a: Barraskewda|89/231
|-unboost|p2a: Barraskewda|atk|1
|
|-heal|p1a: Dachsbun|68/252|[from] item: Leftovers
|upkeep
|turn|11
|
|t:|1711853652
|move|p1a: Dachsbun|Protect|p1a: Dachsbun
|-singleturn|p1a: Dachsbun|Protect
|move|p2a: Barraskewda|Poison Jab|p1a: Dachsbun
|-activate|p1a: Dachsbun|move: Protect
|
|-heal|p1a: Dachsbun|83/252|[from] item: Leftovers
|upkeep
|turn|12
|inactive|actualboy has 120 seconds left.
|
|t:|1711853688
|switch|p1a: Bombirdier|Bombirdier, L85, F|258/258
|move|p2a: Barraskewda|Poison Jab|p1a: Bombirdier
|-damage|p1a: Bombirdier|198/258
|
|upkeep
|turn|13
|inactive|actualboy has 120 seconds left.
|inactive|gaidepcobird has 120 seconds left.
|
|t:|1711853715
|switch|p2a: Regigigas|Regigigas, L84|322/322
|-start|p2a: Regigigas|ability: Slow Start
|move|p1a: Bombirdier|Knock Off|p2a: Regigigas
|-damage|p2a: Regigigas|181/322
|-enditem|p2a: Regigigas|Leftovers|[from] move: Knock Off|[of] p1a: Bombirdier
|
|upkeep
|turn|14
|inactive|gaidepcobird has 120 seconds left.
|inactive|actualboy has 120 seconds left.
|
|t:|1711853727
|switch|p1a: Dachsbun|Dachsbun, L91, M|83/252
|move|p2a: Regigigas|Protect||[still]
|-fail|p2a: Regigigas
|
|-heal|p1a: Dachsbun|98/252|[from] item: Leftovers
|upkeep
|turn|15
|inactive|gaidepcobird has 120 seconds left.
|
|t:|1711853744
|-terastallize|p2a: Regigigas|Poison
|move|p1a: Dachsbun|Body Press|p2a: Regigigas
|-resisted|p2a: Regigigas
|-crit|p2a: Regigigas
|-damage|p2a: Regigigas|133/322
|move|p2a: Regigigas|Substitute|p2a: Regigigas
|-start|p2a: Regigigas|Substitute
|-damage|p2a: Regigigas|53/322
|
|-heal|p1a: Dachsbun|113/252|[from] item: Leftovers
|upkeep
|turn|16
|
|t:|1711853756
|move|p2a: Regigigas|Protect|p2a: Regigigas
|-singleturn|p2a: Regigigas|Protect
|move|p1a: Dachsbun|Play Rough|p2a: Regigigas
|-activate|p2a: Regigigas|move: Protect
|
|-heal|p1a: Dachsbun|128/252|[from] item: Leftovers
|upkeep
|turn|17
|
|t:|1711853770
|move|p1a: Dachsbun|Play Rough|p2a: Regigigas
|-resisted|p2a: Regigigas
|-activate|p2a: Regigigas|move: Substitute|[damage]
|move|p2a: Regigigas|Body Slam|p1a: Dachsbun
|-damage|p1a: Dachsbun|77/252
|
|-heal|p1a: Dachsbun|92/252|[from] item: Leftovers
|-end|p2a: Regigigas|Slow Start
|upkeep
|turn|18
|
|t:|1711853783
|move|p2a: Regigigas|Protect|p2a: Regigigas
|-singleturn|p2a: Regigigas|Protect
|move|p1a: Dachsbun|Play Rough|p2a: Regigigas
|-activate|p2a: Regigigas|move: Protect
|
|-heal|p1a: Dachsbun|107/252|[from] item: Leftovers
|upkeep
|turn|19
|
|t:|1711853793
|move|p1a: Dachsbun|Play Rough|p2a: Regigigas
|-resisted|p2a: Regigigas
|-end|p2a: Regigigas|Substitute
|move|p2a: Regigigas|Body Slam|p1a: Dachsbun
|-damage|p1a: Dachsbun|5/252
|
|-heal|p1a: Dachsbun|20/252|[from] item: Leftovers
|upkeep
|turn|20
|inactive|gaidepcobird has 120 seconds left.
|
|t:|1711853804
|move|p1a: Dachsbun|Play Rough|p2a: Regigigas
|-resisted|p2a: Regigigas
|-damage|p2a: Regigigas|11/322
|move|p2a: Regigigas|Body Slam|p1a: Dachsbun
|-damage|p1a: Dachsbun|0 fnt
|faint|p1a: Dachsbun
|
|upkeep
|
|t:|1711853818
|switch|p1a: Swanna|Swanna, L88, F|275/275
|turn|21
|
|t:|1711853829
|move|p1a: Swanna|Hydro Pump|p2a: Regigigas
|-damage|p2a: Regigigas|0 fnt
|faint|p2a: Regigigas
|-end|p2a: Regigigas|Slow Start|[silent]
|
|upkeep
|inactive|gaidepcobird has 120 seconds left.
|inactive|gaidepcobird has 90 seconds left.
|inactive|gaidepcobird has 60 seconds left.
|inactive|gaidepcobird has 30 seconds left.
|inactive|gaidepcobird has 20 seconds left.
|inactive|gaidepcobird has 15 seconds left.
|inactive|gaidepcobird has 10 seconds left.
|inactive|gaidepcobird has 5 seconds left.
|-message|gaidepcobird lost due to inactivity.
|
|win|actualboy
|raw|actualboy's rating: 1152 &rarr; <strong>1178</strong><br />(+26 for winning)
|raw|gaidepcobird's rating: 1168 &rarr; <strong>1142</strong><br />(-26 for losing)
|l|☆actualboy
|player|p1|
"""

def main():
    global game_log
    filename = 'b.json'
    if len(sys.argv) > 1:
        game_log = requests.get(sys.argv[1]).text  
        filename = sys.argv[1].split('-')[1].replace('log', 'json')
    game_log = game_log.replace('|start', '|turn|0')
    # Parse the log
    # Parse the game log
    detailed_dataset = []
    for turn in game_log.split('|turn|')[1:]:
        if len(detailed_dataset) == 0:
            last_turn_state = {"p1a": {}, "p2a": {}} 
        else:
            last_turn_state = detailed_dataset[-1]["post_action_state"]
        detailed_dataset.extend(parse_detailed_turn(turn, last_turn_state))

    # For demonstration, print the structured dataset
    #pprint(detailed_dataset)
    for d in detailed_dataset:
        print(json.dumps(d, indent=2))
    with open(filename, 'w+') as f:
        json.dump(detailed_dataset, f, indent=2)

if __name__ == '__main__':
    main()