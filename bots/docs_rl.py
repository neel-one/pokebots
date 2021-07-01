from poke_env.player.env_player import Gen8EnvSinglePlayer
import numpy as np
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential, load_model
from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
from tensorflow.keras.optimizers import Adam
from greedy import Greedy
from model_player import ModelPlayer

NB_TRAINING_STEPS = 10000
NB_EVALUATION_EPISODES = 100
class Elite4(Gen8EnvSinglePlayer):
    def embed_battle(self, battle):
        # -1 indicates that the move does not have a base power
        # or is not available
        moves_base_power = -np.ones(4)
        moves_dmg_multiplier = np.ones(4)
        for i, move in enumerate(battle.available_moves):
            moves_base_power[i] = (
                move.base_power / 100
            )  # Simple rescaling to facilitate learning
            if move.type:
                moves_dmg_multiplier[i] = move.type.damage_multiplier(
                    battle.opponent_active_pokemon.type_1,
                    battle.opponent_active_pokemon.type_2,
                )
       
        canDyna = 0
        if battle.can_dynamax:
            canDyna = 1
            
        canODyna = 0
        if battle.opponent_can_dynamax:
            canODyna = 1
        
        dynamaxTurnsLeft = battle.dynamax_turns_left
        if battle.dynamax_turns_left == None:
            dynamaxTurnsLeft = -1
            
        OdynamaxTurnsLeft = battle.opponent_dynamax_turns_left
        if battle.opponent_dynamax_turns_left == None:
            OdynamaxTurnsLeft = -1
        # We count how many pokemons have not fainted in each team
        remaining_mon_team = (
            len([mon for mon in battle.team.values() if mon.fainted]) / 6
        )
        remaining_mon_opponent = (
            len([mon for mon in battle.opponent_team.values() if mon.fainted]) / 6
        )
        
        our_cur_status = -1
        if battle.active_pokemon.status:
            our_cur_status = battle.active_pokemon.status.value/7
        
        opp_cur_status = -1
        if battle.opponent_active_pokemon.status:
            opp_cur_status = battle.opponent_active_pokemon.status.value/7
            
        our_cur_types = -1*np.ones(2)
        opp_cur_types = -1*np.ones(2)
        if battle.active_pokemon:
            our_cur_types[0] = battle.active_pokemon.type_1.value/18
            if battle.active_pokemon.type_2:
                our_cur_types[1] = battle.active_pokemon.type_2.value/18
        
        if battle.active_pokemon:
            opp_cur_types[0] = battle.opponent_active_pokemon.type_1.value/18
            if battle.opponent_active_pokemon.type_2:
                opp_cur_types[1] = battle.opponent_active_pokemon.type_2.value/18
        
        firstturn = battle.active_pokemon.first_turn
        
        # Final vector with 10 components
        return np.concatenate(
            [
                moves_base_power,
                moves_dmg_multiplier,
                [remaining_mon_team, remaining_mon_opponent],
                [canDyna,
                canODyna,
                firstturn,
                dynamaxTurnsLeft,
                OdynamaxTurnsLeft,
                our_cur_status,
                opp_cur_status],
                our_cur_types,
                opp_cur_types
            ]
        )

    def compute_reward(self, battle) -> float:
        return self.reward_computing_helper(
            battle, fainted_value=1, hp_value=1, victory_value=30
        )

class SimpleRLPlayer(Gen8EnvSinglePlayer):
    def embed_battle(self, battle):
        # -1 indicates that the move does not have a base power
        # or is not available
        moves_base_power = -np.ones(4)
        moves_dmg_multiplier = np.ones(4)
        for i, move in enumerate(battle.available_moves):
            moves_base_power[i] = (
                move.base_power / 100
            )  # Simple rescaling to facilitate learning
            if move.type:
                moves_dmg_multiplier[i] = move.type.damage_multiplier(
                    battle.opponent_active_pokemon.type_1,
                    battle.opponent_active_pokemon.type_2,
                )

        # We count how many pokemons have not fainted in each team
        remaining_mon_team = (
            len([mon for mon in battle.team.values() if mon.fainted]) / 6
        )
        remaining_mon_opponent = (
            len([mon for mon in battle.opponent_team.values() if mon.fainted]) / 6
        )

        # Final vector with 10 components
        return np.concatenate(
            [
                moves_base_power,
                moves_dmg_multiplier,
                [remaining_mon_team, remaining_mon_opponent],
            ]
        )

    def compute_reward(self, battle) -> float:
        return self.reward_computing_helper(
            battle, fainted_value=2, hp_value=1, victory_value=30
        )

def dqn_training(player, dqn, nb_steps):
    dqn.fit(player, nb_steps=nb_steps)
    player.complete_current_battle()

def dqn_evaluation(player, dqn, nb_episodes):
    player.reset_battles()
    dqn.test(player, nb_episodes=nb_episodes, visualize=False, verbose=False)

    print(
        "DQN Evaluation: %d victories out of %d episodes"
        % (player.n_won_battles, nb_episodes)
    )
def create_model(n_action=None,load=None):
    if load is not None:
        pass

if __name__=='__main__':
    env_player = SimpleRLPlayer(battle_format="gen8randombattle")

    opponent = Greedy(battle_format="gen8randombattle")

    # Output dimension
    n_action = len(env_player.action_space)

    model = Sequential()
    model.add(Dense(128, activation="elu", input_shape=(1, 10)))

    # Our embedding have shape (1, 10), which affects our hidden layer
    # dimension and output dimension
    # Flattening resolve potential issues that would arise otherwise
    model.add(Flatten())
    model.add(Dense(64, activation="elu"))
    model.add(Dense(n_action, activation="linear"))

    memory = SequentialMemory(limit=10000, window_length=1)

    # Ssimple epsilon greedy
    policy = LinearAnnealedPolicy(
        EpsGreedyQPolicy(),
        attr="eps",
        value_max=1.0,
        value_min=0.05,
        value_test=0,
        nb_steps=10000,
    )

    # Defining our DQN
    dqn = DQNAgent(
        model=model,
        nb_actions=len(env_player.action_space),
        policy=policy,
        memory=memory,
        nb_steps_warmup=1000,
        gamma=0.5,
        target_model_update=1,
        delta_clip=0.01,
        enable_double_dqn=True,
    )

    dqn.compile(Adam(lr=0.00025), metrics=["mae"])
    
    opponent = ModelPlayer(load_model('MDSTPoke/model_20000'), Elite4())
    # Training
    NUM_EPOCHS = 37
    steps = 100 #NB_TRAINING_STEPS*NUM_EPOCHS 
    env_player.play_against(
        env_algorithm=dqn_training,
        opponent=opponent,
        env_algorithm_kwargs={"dqn": dqn, "nb_steps": steps},
    )

    #model.save("model_%d" % (NB_TRAINING_STEPS*NUM_EPOCHS))
    #also have to serialize memory?

    #opponent = ModelPlayer(load_model('models/model_370000'), env_player)
    # Evaluation
    print("Results against Greedy:")
    env_player.play_against(
        env_algorithm=dqn_evaluation,
        opponent=opponent,
        env_algorithm_kwargs={"dqn": dqn, "nb_episodes": NB_EVALUATION_EPISODES},
    )