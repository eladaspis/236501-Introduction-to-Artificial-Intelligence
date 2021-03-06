"""Search Algos: MiniMax, AlphaBeta
"""
from utils import ALPHA_VALUE_INIT, BETA_VALUE_INIT
#TODO: you can import more modules, if needed
import numpy as np


class SearchAlgos:
    def __init__(self, utility, succ, perform_move, goal=None):
        """The constructor for all the search algos.
        You can code these functions as you like to, 
        and use them in MiniMax and AlphaBeta algos as learned in class
        :param utility: The utility function.
        :param succ: The succesor function.
        :param perform_move: The perform move function.
        :param goal: function that check if you are in a goal state.
        """
        self.utility = utility
        self.succ = succ
        self.perform_move = perform_move
        self.depth = 0

def search(self, state, depth, maximizing_player):
        pass

def my_print(board):
    board = np.flipud(board)
    print('_' * len(board[0]) * 4)
    for row in board:
        row = [str(int(x)) if x != -1 else 'X' for x in row]
        print(' | '.join(row))
        print('_' * len(row) * 4)
class MiniMax(SearchAlgos):

    def search(self, state, depth, maximizing_player):
        """Start the MiniMax algorithm.
        :param state: The state to start from.
        :param depth: The maximum allowed depth for the algorithm.
        :param maximizing_player: Whether this is a max node (True) or a min node (False).
        :return: A tuple: (The min max algorithm value, The direction in case of max node or None in min mode)
        """
        if state.no_more_moves(2 - int(maximizing_player)):
            return state.utility(2 - int(maximizing_player)), state.get_player_pos(1)
        if depth == 0:
            tmp = state.heuiristic(state.board)
            return tmp, state.get_player_pos(1)

        CurMax = float('-inf')
        CurMin = float('inf')
        best_move = None
        if maximizing_player:                        #max Node
            for c in state.succ(state.get_player_pos(2-int(maximizing_player))):
                old_pos1 = state.get_player_pos(2-int(maximizing_player))
                next_tile_value1 = state.board[c]
                self.depth += 1
                state.sum_fruit += (1/self.depth)*next_tile_value1
                state.perform_move(c, 2-int(maximizing_player))
                v = self.search(state, depth-1, not maximizing_player)
                state.undo_move(old_pos1, 2-int(maximizing_player), next_tile_value1)
                state.sum_fruit -= (1/self.depth)*next_tile_value1
                self.depth -= 1
                if CurMax < v[0]:
                    CurMax = v[0]
                    best_move = c
            return CurMax, best_move
        else:     

            for c in state.succ(state.get_player_pos(2-int(maximizing_player))):        #Min Node
                old_pos2 = state.get_player_pos(2 - int(maximizing_player))
                next_tile_value2 = state.board[c]
                state.perform_move(c, 2-int(maximizing_player))
                v = self.search(state, depth-1, not maximizing_player)
                state.undo_move(old_pos2, 2 - int(maximizing_player), next_tile_value2)
                if CurMin > v[0]:
                    CurMin = v[0]
                    best_move = c
            return CurMin, best_move
        


class AlphaBeta(SearchAlgos):

    def search(self, state, depth, maximizing_player, alpha=ALPHA_VALUE_INIT, beta=BETA_VALUE_INIT):
        """Start the AlphaBeta algorithm.
        :param state: The state to start from.
        :param depth: The maximum allowed depth for the algorithm.
        :param maximizing_player: Whether this is a max node (True) or a min node (False).
        :param alpha: alpha value
        :param: beta: beta value
        :return: A tuple: (The min max algorithm value, The direction in case of max node or None in min mode)
        """

        if state.no_more_moves(2 - int(maximizing_player)):
            return state.utility(2 - int(maximizing_player)), state.get_player_pos(1)
        if depth == 0:
            tmp = state.heuiristic(state.board)
            return tmp, state.get_player_pos(1)

        CurMax = float('-inf')
        CurMin = float('inf')
        best_move = None
        if maximizing_player:  # max Node
            for c in state.succ(state.get_player_pos(2 - int(maximizing_player))):
                old_pos1 = state.get_player_pos(2 - int(maximizing_player))
                next_tile_value1 = state.board[c]
                self.depth += 1
                state.sum_fruit += (1/self.depth)*next_tile_value1
                state.perform_move(c, 2 - int(maximizing_player))
                v = self.search(state, depth - 1, not maximizing_player, alpha, beta)
                state.undo_move(old_pos1, 2 - int(maximizing_player), next_tile_value1)
                state.sum_fruit -= (1/self.depth)*next_tile_value1
                self.depth -= 1
                if CurMax < v[0]:
                    CurMax = v[0]
                    best_move = c
                alpha = max(CurMax, alpha)
                if CurMax >= beta:
                    return float('inf'), c
            return CurMax, best_move
        else:                                                      #Min Node

            for c in state.succ(state.get_player_pos(2 - int(maximizing_player))):
                old_pos2 = state.get_player_pos(2 - int(maximizing_player))
                next_tile_value2 = state.board[c]
                state.perform_move(c, 2 - int(maximizing_player))
                v = self.search(state, depth - 1, not maximizing_player, alpha, beta)
                state.undo_move(old_pos2, 2 - int(maximizing_player), next_tile_value2)
                if CurMin > v[0]:
                    CurMin = v[0]
                    best_move = c

                beta = min(CurMin, beta)
                if CurMin <= alpha:
                    return float('-inf'), c
            return CurMin, best_move
