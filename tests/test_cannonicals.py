import numpy as np

from soulaween.SoulaweenGame import SoulaweenGame


g = SoulaweenGame(4)

board = np.array([[0] * 4] * 4)
board[0,0] = 1
board[0,1] = 1
board[0,2] = -1
board[3, 1] = -1
pi = np.array([[[0] * 4] * 4] * 2)

can = g.getCanonicalForm(board, 1)

symmetries = g.getSymmetries(can, pi.flatten())
assert all([np.array_equal(can, g.getCanonicalForm(b, 1)) for b, _ in symmetries])
print("All symmetries' canonicals are the same!")

# ### Test reverseTransformation

board = np.arange(16).reshape(4, 4)
pi = np.stack([board, -board])
print("Board:")
print(board)

canonicalBoard, idx = g.getCanonicalForm(board, 1, True)
print("Canonical board:")
print(canonicalBoard)

i = idx // 4  # Rotation steps (0, 1, 2, 3)
idx = idx % 4
j = 1 if idx // 2 == 0 else 0 # Flip (1 or 0)
idx = idx % 2
k = -1 if idx == 1 else 1  # Color inversion
print(f"Transformation index: {idx} ->", i, j, k)


reversed_board = np.copy(canonicalBoard)
if j:
    reversed_board = np.flip(reversed_board, axis=1)
# Reverse the rotation
reversed_board = np.rot90(reversed_board, -i) * k
print("Reversed board:")
print(reversed_board)



### Test MCTS
from MCTS import MCTS
from soulaween.pytorch.NNet import NNetWrapper as nn
from utils import *

args = dotdict(
    {
        "numIters": 1000,
        "numEps": 100,  # Number of complete self-play games to simulate during a new iteration.
        "tempThreshold": 15,  #
        "updateThreshold": 0.6,  # During arena playoff, new neural net will be accepted if threshold or more of games are won.
        "maxlenOfQueue": 200000,  # Number of game examples to train the neural networks.
        "numMCTSSims": 25,  # Number of games moves for MCTS to simulate.
        "arenaCompare": 40,  # Number of games to play during arena play to determine if new net will be accepted.
        "cpuct": 1,
        "checkpoint": "./temp/",
        "load_model": False,
        "load_folder_file": ("./temp/", "best.pth.tar"),
        "numItersForTrainExamplesHistory": 20,
    }
)

nnet = nn(g)
mcts = MCTS(g, nnet, args)
board = g.getInitBoard()
curPlayer = 1
episodeStep = 0

while True:
    episodeStep += 1
    canonicalBoard, idx = g.getCanonicalForm(board, curPlayer, True)
    temp = int(episodeStep < args.tempThreshold)

    pi = np.array(mcts.getActionProb(canonicalBoard, temp=temp))
    sym = g.getSymmetries(canonicalBoard, pi)

    reversed_pi = g.reverseTransformation(pi, idx)
    action = np.random.choice(len(reversed_pi), p=reversed_pi)
    board, curPlayer = g.getNextState(board, curPlayer, action)
    r = g.getGameEnded(board, curPlayer)
    if r != 0:
        break
print("MCTS works!")