import sys

sys.path.append("..")
# from Game import Game
from .BoopLogic import Board
import numpy as np

class BoopGame():
    """
    This class specifies the game Boop. 

    Use 1 for player1 and -1 for player2.
    """
    def __init__(self, n):
        self.n = n

    def getInitBoard(self):
        """
        Returns:
            startBoard: a representation of the board
        """
        b = Board(self.n)
        return np.array(b.pieces)

    def getBoardSize(self):
        """
        Returns:
            (x,y): a tuple of board dimensions
        """
        return (self.n + 1, self.n)

    def getActionSize(self):
        """
        Returns:
            actionSize: number of all possible actions
        """
        return 2*self.n*self.n + 1

    def getNextState(self, board, player, action):
        """
        Input:
            board: current board
            player: current player (1 or -1)
            action: action taken by current player

        Returns:
            nextBoard: board after applying action
            nextPlayer: player who plays in the next turn (should be -player)
        """
        b = Board(self.n)
        b.pieces = np.copy(board)
        if action == self.n*self.n:
            assert any(b.pieces[:-1].flatten() == player)
            first_kitten = np.where(b.pieces[:-1] == player)
            b._remove_piece(first_kitten[0][0], first_kitten[1][0])
        else:
            piece = 1 if action // (self.n**2) == 0 else 2
            action = action % (self.n**2)
            move = (int(action/self.n), action%self.n)
            b.execute_move(move, piece * player)
        return (b.pieces, -player)

    def getValidMoves(self, board, player):
        """
        Input:
            board: current board
            player: current player

        Returns:
            validMoves: a binary vector of length self.getActionSize(), 1 for
                        moves that are valid from the current board and player,
                        0 for invalid moves
        """
        valids = [0] * self.getActionSize()
        b = Board(self.n)
        b.pieces = np.copy(board)
        freeSquares = b.get_free_squares()
        for x, y in freeSquares:
            if b.pieces[-1][-(player - 1)] > 0:
                valids[self.n * x + y] = 1
            if b.pieces[-1][-(player - 1) + 1] > 0:
                valids[self.n * x + y + self.n * self.n] = 1
        if not any(valids):
            valids[-1] = 1
        return np.array(valids)

    def getGameEnded(self, board, player):
        """
        Input:
            board: current board
            player: current player (1 or -1)

        Returns:
            r: 0 if game has not ended. 1 if player won, -1 if player lost,
               small non-zero value for draw.
               
        """
        b = Board(self.n)
        b.pieces = np.copy(board)
        return b.is_win(player)

    def getCanonicalForm(self, board, player):
        """
        Input:
            board: current board
            player: current player (1 or -1)

        Returns:
            canonicalBoard: returns canonical form of board. The canonical form
                            should be independent of player. For e.g. in chess,
                            the canonical form can be chosen to be from the pov
                            of white. When the player is white, we can return
                            board as is. When the player is black, we can invert
                            the colors and return the board.
        """
        # change values of last row according to indices (0, 1, 2, 3) -> (2, 3, 0, 1)
        board[-1] = np.roll(board[-1], 2*(player - 1))
        board[:-1] = player*board[:-1]
        return board

    def getSymmetries(self, board, pi):
        """
        Input:
            board: current board
            pi: policy vector of size self.getActionSize()

        Returns:
            symmForms: a list of [(board,pi)] where each tuple is a symmetrical
                       form of the board and the corresponding pi vector. This
                       is used when training the neural network from examples.
        """
        assert(len(pi) == self.n**2+1)  # 1 for pass
        pi_board = np.reshape(pi[:-1], (2, self.n, self.n))
        l = []

        for i in range(1, 5):
            for j in [True, False]:
                newB = np.rot90(board[:-1], i)
                newPi = np.rot90(pi_board, i, axes=(1, 2))
                if j:
                    newB = np.fliplr(newB)
                    newPi = np.fliplr(newPi)
                l += [(np.concatenate([newB, board[-1]]), list(newPi.ravel()) + [pi[-1]])]
        return l

    def stringRepresentation(self, board):
        """
        Input:
            board: current board

        Returns:
            boardString: a quick conversion of board to a string format.
                         Required by MCTS for hashing.
        """
        return board.tostring()
    
    @staticmethod
    def display(board):
        n = board.shape[1]
        print("   ", end="")
        for y in range(n):
            print(y, end=" ")
        print("")
        print("-----------------------")
        for y in range(n):
            print(y, "|", end="")    # print the row #
            for x in range(n):
                piece = board[y][x]
                if piece == 1:
                    print("x", end=" ")
                elif piece == 2:
                    print("X", end=" ")
                elif piece == -1:
                    print("o", end=" ")
                elif piece == -2:
                    print("O", end=" ")
                else:
                    print(".", end=" ")
            print("|")
        print("-----------------------")
        print(f"Pieces: {board[-1][:-2]}")