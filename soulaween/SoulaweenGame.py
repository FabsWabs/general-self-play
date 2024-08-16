import sys

sys.path.append("..")
# from Game import Game
from .SoulaweenLogic import Board
import numpy as np


class SoulaweenGame:
    def __init__(self, n=4) -> None:
        self.n = n

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
            winner = np.count_nonzero(b.pieces) % 2 # 0 if player 2 won, 1 if player 1 won
            if winner == 0:
                winner = -1
            if winner == player:
                return 1
            else:
                return -1
        if not b.has_legal_moves():
            return 1e-4
        return 0
        
    def getCanonicalForm(self, board, player, return_index=False):
        """Find the canonical form of the board (lexicographically smallest)."""
        symmetries = self.getSymmetries(board, np.zeros(self.n**2 * 2))
        symmetries_as_tuples = [tuple(sym.flat) for sym, _ in symmetries]
        
        # Find the canonical tuple
        canonical_tuple = min(symmetries_as_tuples)
        canonical_index = symmetries_as_tuples.index(canonical_tuple)  # Get the correct index
        size = board.shape[0]
        canonical_board = np.array(canonical_tuple).reshape(size, size)
    
        if return_index:
            return canonical_board, canonical_index
        else:
            return canonical_board
    
    def reverseTransformation(self, action_or_pi_board, transformation_index):
        """Reverses the transformation to map the flattened policy board or integer action
        back to the original board's action space."""
    
        # Extract the original transformation parameters
        i = transformation_index // 4  # Rotation steps (0, 1, 2, 3)
        transformation_index = transformation_index % 4
        j = 1 if transformation_index // 2 == 0 else 0  # Flip (1 or 0)
        transformation_index = transformation_index % 2
        k = -1 if transformation_index == 1 else 1  # Color inversion

        if isinstance(action_or_pi_board, np.integer):
            # If input is an integer action
            pi_board = np.zeros((2, self.n, self.n))
            pi_board.ravel()[action_or_pi_board] = 1
        else:
            # If input is a flattened pi_board
            pi_board = np.array(action_or_pi_board).reshape(2, self.n, self.n)

        # Reverse the flip
        if j:
            pi_board = np.flip(pi_board, axis=2)

        # Reverse the rotation
        pi_board = np.rot90(pi_board, -i, axes=(1, 2))

        # Reverse the color inversion
        pi_board = pi_board[::k, :, :]

        # Flatten the transformed board
        flattened_board = pi_board.ravel()

        if isinstance(action_or_pi_board, np.integer):
            # Return the index of the maximum action in the original board's action space
            return np.argmax(flattened_board)
        else:
            # Return the flattened transformed policy board
            return flattened_board

    def getSymmetries(self, board, pi):
        # mirror, rotational
        assert len(pi) == self.n**2 * 2
        pi_board = np.reshape(pi, (2, self.n, self.n))
        l = []

        for i in range(0, 4):
            for j in [True, False]:
                for k in [1, -1]:
                    newB = np.rot90(board, i) * k
                    newPi = np.rot90(pi_board, i, axes=(1, 2))[::k, :, :]
                    if j:
                        newB = np.fliplr(newB)
                        newPi = np.flip(newPi, axis=2)
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
