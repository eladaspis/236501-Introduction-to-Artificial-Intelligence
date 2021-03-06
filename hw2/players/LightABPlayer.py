"""
MiniMax Player with AlphaBeta pruning with light heuristic
"""

import numpy as np
import SearchAlgos
from players.AbstractPlayer import AbstractPlayer
#TODO: you can import more modules, if needed


PLAYER_ONE = 1
PLAYER_TWO = 2

class Player(AbstractPlayer):
    def __init__(self, game_time, penalty_score):
        AbstractPlayer.__init__(self, game_time, penalty_score) # keep the inheritance of the parent's (AbstractPlayer) __init__()
        self.board = None
        self.pos = None
        self.alpha_beta = SearchAlgos.AlphaBeta(self.utility, self.succ, self.make_move)
        self.fruit = None
        self.my_score = 0
        self.rival_score = 0
        self.sum_fruit = 0


    def set_game_params(self, board):
        """Set the game parameters needed for this player.
        This function is called before the game starts.
        (See GameWrapper.py for more info where it is called)
        input:
            - board: np.array, a 2D matrix of the board.
        No output is expected.
        """
        self.board = board
        pos = np.where(board == 1)
        # convert pos to tuple of ints
        self.pos = tuple(ax[0] for ax in pos)

    @staticmethod
    def count_ones(board):
        counter = len(np.where(board == 1)[0])
        return counter

    def make_move(self, time_limit, players_score):
        """Make move with this Player.
        input:
            - time_limit: float, time limit for a single turn.
        output:
            - direction: tuple, specifing the Player's movement, chosen from self.directions
        """
        prev_pos = self.pos
        assert self.count_ones(self.board) == 1
        next_pos = None
        min_max_value, next_pos = self.alpha_beta.search(self, 4, True, float('-inf'), float('inf'))
        i = int(next_pos[0] - prev_pos[0])
        j = int(next_pos[1] - prev_pos[1])
        self.board[prev_pos] = -1
        self.board[next_pos] = 1
        self.pos = next_pos
        return (i, j)


    def set_rival_move(self, pos):
        """Update your info, given the new position of the rival.
        input:
            - pos: tuple, the new position of the rival.
        No output is expected
        """
        old_pos = np.where(self.board == 2)
        if old_pos is not None:
            self.board[old_pos] = -1
        self.board[pos] = 2



    def update_fruits(self, fruits_on_board_dict):
        """Update your info on the current fruits on board (if needed).
        input:
            - fruits_on_board_dict: dict of {pos: value}
                                    where 'pos' is a tuple describing the fruit's position on board,
                                    'value' is the value of this fruit.
        No output is expected.
        """
        for e in fruits_on_board_dict:
            if self.board[e] is not ['-1', '1', '2']:
                self.board[e] = fruits_on_board_dict[e]
        self.fruits = fruits_on_board_dict

    def succ(self, pos):
        new_list = []
        for d in self.directions:  # (1,2)
            i = pos[0] + d[0]
            j = pos[1] + d[1]
            if 0 <= i < len(self.board) and 0 <= j < len(self.board[0]) and (
                    self.board[int(i)][int(j)] not in [-1, 1, 2]):  # then move is legal
                new_list.append((int(i), int(j)))

        return new_list

    def get_player_pos(self, player_number):
        pos = np.where(self.board == player_number)
        # convert pos to tuple of ints
        return pos

    def enemyDistance(self, board):
        player_1 = self.get_player_pos(1)
        player_2 = self.get_player_pos(2)
        diff_x = abs(player_1[0] - player_2[0])
        diff_y = abs(player_1[1] - player_2[1])
        return int(diff_x + diff_y)

    def minDistToFruit(self, board, pos):
        best_distance = float('inf')
        for i in range(len(board)):
            for j in range(len(board[0])):
                if board[i][j] not in [-1, 1, 2, 0]:
                    diff_x = abs(pos[0] - i)
                    diff_y = abs(pos[1] - j)
                    dist = diff_x + diff_y
                    if dist < best_distance:
                        best_distance = dist
        return best_distance

    def availableMoves(self, board, pos):
        available_moves = 0
        for d in self.directions:
            i = pos[0] + d[0]
            j = pos[1] + d[1]
            if 0 <= i < len(board) and 0 <= j < len(board[0]) and board[int(i)][int(j)] == 0:  # then move is legal
                available_moves += 1
        return available_moves

    def minDistFromFrame(self, board, pos):
        x = min(len(board) - pos[0], 1 + pos[0])
        y = min(len(board[0]) - pos[1], 1 + pos[1])
        return min(x, y)

    def state_score(self, board, pos):
        num_steps_available = 0
        for d in self.directions:
            i = pos[0] + d[0]
            j = pos[1] + d[1]
            # check legal move
            if 0 <= i < len(board) and 0 <= j < len(board[0]) and (board[int(i)][int(j)] not in [-1, 1, 2]):
                num_steps_available += 1

        if num_steps_available == 0:
            return -1
        else:
            return 4 - num_steps_available

    def heuiristic(self, board):
        curr_pos = self.get_player_pos(PLAYER_ONE)
        simple_heur = self.state_score(board, curr_pos)
        dist_to_frame = 1/self.minDistFromFrame(board, curr_pos)
        res = simple_heur + dist_to_frame
        return res

    def utility(self, player_number):
        pos1 = self.get_player_pos(player_number)
        pos2 = self.get_player_pos(3-player_number)
        neighbors_1 = self.succ(pos1)
        neighbors_2 = self.succ(pos2)

        if len(neighbors_1) == 0 or len(neighbors_2) == 0:  # player 1 and 2 cant move
            if len(neighbors_2) > 0: # player 1 cannot move
                if self.my_score - self.penalty_score > self.rival_score:
                    return 1
                elif self.my_score - self.penalty_score < self.rival_score:
                    return -1
                else:
                    return 0
            if len(neighbors_1) > 0: # player 2 cannot move
                if self.my_score > self.rival_score - self.penalty_score:
                    return 1
                elif self.my_score < self.rival_score - self.penalty_score:
                    return -1
                else:
                    return 0
            if self.my_score > self.rival_score:
                return 1
            elif self.my_score < self.rival_score:
                return -1
            else:
                return 0
        return 0

    def perform_move(self, new_pos, player_number):
        if self.board[new_pos] not in [-1, 3 - player_number]:
            self.board[self.get_player_pos(player_number)] = -1
            self.board[new_pos] = player_number
            self.pos = new_pos
            if player_number == PLAYER_ONE and self.board[new_pos] > 2:
                self.my_score += self.board[new_pos]
            if player_number == PLAYER_TWO and self.board[new_pos] > 2:
                self.rival_score += self.board[new_pos]

    def undo_move(self, old_pos, player_number, old_value):
        self.board[self.get_player_pos(player_number)] = old_value
        self.board[old_pos] = player_number
        self.pos = old_pos
        if old_value > 2:
            if player_number == PLAYER_ONE:
                self.my_score -= old_value
            else:
                self.rival_score -= old_value

    def no_more_moves(self, player_number):
        pos1 = self.get_player_pos(PLAYER_ONE)
        pos2 = self.get_player_pos(PLAYER_TWO)
        neighbors_1 = self.succ(pos1)
        neighbors_2 = self.succ(pos2)
        if len(neighbors_1) == 0 and len(neighbors_2) == 0:  # player 1 and 2 cant move
            return True
        if player_number == PLAYER_ONE:
            if len(neighbors_1) == 0 and len(neighbors_2) != 0:  # player 1 lose
                return True
            if len(neighbors_1) != 0 and len(neighbors_2) == 0:  #
                return False
        if player_number == PLAYER_TWO:
            if len(neighbors_1) == 0 and len(neighbors_2) != 0:  #
                return False
            if len(neighbors_1) != 0 and len(neighbors_2) == 0:  #
                return True
        return False
