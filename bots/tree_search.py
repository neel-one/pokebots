import poke-env
import numpy as np

def move_to_array(move):
    pass


def pokemon_to_array(pokemon):
    pass


def battle_to_state(battle):
    '''
    Convert battle object from poke-env to a state array
    End goal -- create a user defined classes to represent state
    '''
    pass

class BaseState:
    '''
    State value must must a 1D array

    class State(BaseState):
        def serialize(self, battle):
            #define serialize method

    s = State()
    battle = getBattleObject()
    state = s(battle)
    '''
    def serialize(self, battle):
        '''
        Return a numpy array representing state from a battle object
        '''
        raise NotImplementedError('serealize must be implemented to represent state')

    def __call__(self, battle):
        return self.serialize(battle)

class Node:
    '''
    State representation - [current_battle_properties,
    '''
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.is_terminal = False
        self.turn = not self.parent.turn if self.parent is not None else True



    def generateChildren(self):
        pass
