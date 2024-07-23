import numpy as np

from .SoulaweenLogic import Board

"""
Random, Greedy and Human-interacting players for the game of Soulaween.

"""


class RandomPlayer:
    def __init__(self, game):
        self.game = game

    def play(self, board):
        valid_actions = np.argwhere(self.game.getValidMoves(board, 1) == 1).flatten()
        return np.random.choice(valid_actions)


class OneStepLookaheadPlayer:
    def __init__(self, game):
        self.game = game

    def play(self, board):
        valids = self.game.getValidMoves(board, 1)
        legal_actions = np.argwhere(valids == 1).flatten()
        for action in legal_actions:
            b = Board(self.game.n)
            b.pieces = np.copy(board)
            color = 1 if action // (self.game.n**2) == 0 else -1
            action = action % (self.game.n**2)
            move = (int(action / self.game.n), action % self.game.n)
            b.execute_move(move, color)
            if b.is_win():
                return action
        valid_actions = np.argwhere(valids == 1).flatten()
        return np.random.choice(valid_actions)


class HumanSoulaweenPlayer:
    def __init__(self, game):
        self.game = game

    def play(self, board):
        # display(board)
        valid = self.game.getValidMoves(board, 1)
        for i in range(len(valid) // 2):
            if valid[i]:
                print(int(i / self.game.n), int(i % self.game.n))
        while True:
            idx, color = input().split(",")
            color = 0 if color.lstrip() in ["X", "1"] else 1
            x, y = [int(x) for x in idx.split(" ")]
            a = self.game.n * x + y + self.game.n**2 * color
            if valid[a]:
                break
            else:
                print("Invalid")

        return a
