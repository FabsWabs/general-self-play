import sys

sys.path.append("..")
# from Game import Game
from .SoulaweenLogic import Board
import numpy as np


class SoulaweenGame:
    def __init__(self, n=4) -> None:
        self.n = n
        self.last_player = -1

    def getInitBoard(self):
        # return initial board (numpy board)
        b = Board(self.n)
        return np.array(b.pieces)

    def getBoardSize(self):
        # (a,b) tuple
        return (self.n, self.n)

    def getActionSize(self):
        # return number of actions
        return self.n * self.n * 2

    def getNextState(self, board, player, action):
        # if player takes action on board, return next (board,player)
        b = Board(self.n)
        b.pieces = np.copy(board)
        color = 1 if action // (self.n**2) == 0 else -1
        action = action % (self.n**2)
        move = (int(action / self.n), action % self.n)
        b.execute_move(move, color)
        self.last_player = player
        return (b.pieces, -player)

    def getValidMoves(self, board, player):
        # return a fixed size binary vector
        valids = [0] * self.getActionSize()
        b = Board(self.n)
        b.pieces = np.copy(board)
        legalMoves = b.get_legal_moves()
        assert len(legalMoves) != 0
        for x, y in legalMoves:
            valids[self.n * x + y] = 1
            valids[self.n * x + y + self.n * self.n] = 1
        return np.array(valids)

    def getGameEnded(self, board, player):
        # return 0 if not ended, 1 if player 1 won, -1 if player 1 lost
        b = Board(self.n)
        b.pieces = np.copy(board)

        if b.is_win():
            return 1 if player == self.last_player else -1
        if not b.has_legal_moves():
            return 1e-4
        return 0

    def getCanonicalForm(self, board, player):
        # return state if player==1, else return -state if player==-1
        return player * board

    def getSymmetries(self, board, pi):
        # mirror, rotational
        assert len(pi) == self.n**2 * 2
        pi_board = np.reshape(pi, (2, self.n, self.n))
        l = []

        for i in range(1, 5):
            for j in [True, False]:
                newB = np.rot90(board, i)
                newPi = np.rot90(pi_board, i, axes=(1, 2))
                if j:
                    newB = np.fliplr(newB)
                    newPi = np.fliplr(newPi)
                l += [(newB, list(newPi.ravel()))]
        return l

    def stringRepresentation(self, board):
        return board.tostring()

    @staticmethod
    def display(board):
        n = board.shape[0]
        print("-------------------------------------------------")
        board_str = "    0   1   2   3\n"
        board_str += "  ┌───┬───┬───┬───┐\n"
        for i in range(n):
            board_str += f"{i} "
            for j in range(n):
                if board[i][j] == 1:
                    board_str += "│ X "
                elif board[i][j] == -1:
                    board_str += "│ O "
                else:
                    board_str += "│ . "
            board_str += "│"
            board_str += "\n"

            if i < n - 1:
                board_str += "  ├───┼───┼───┼───┤\n"
        board_str += "  └───┴───┴───┴───┘"
        print(board_str)
