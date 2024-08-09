import numpy as np
import sys


class RandomPlayer:
    def __init__(self, game):
        self.game = game

    def play(self, board):
        valid_actions = np.argwhere(self.game.getValidMoves(board, 1) == 1).flatten()
        print(valid_actions)
        print(len(valid_actions))
        return np.random.choice(valid_actions)

class HumanBoopPlayer:
    def __init__(self, game):
        self.game = game

    def play(self, board):
        # input as "x y, c" with c being 0 for kitten and 1 for cat
        valid = self.game.getValidMoves(board, 1)
        while True:
            try:
                idx, cat = input().split(",")
                cat = 1 if cat.lstrip() == "c" else 0
                x, y = [int(x) for x in idx.split(" ")]
                a = self.game.n * x + y + self.game.n**2 * cat
                if valid[a]:
                    break
                else:
                    print("Invalid")
            except KeyboardInterrupt:
                sys.exit()
            except:
                print("Invalid")
        return a
