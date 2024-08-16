import numpy as np

from soulaween.SoulaweenGame import SoulaweenGame

g = SoulaweenGame(4)

board = np.array([[0] * 4] * 4)
board[0,0] = 1
board[0,1] = 1
board[0,2] = -1

pi = np.array([[[0] * 4] * 4] * 2)
pi[0, 0, 3] = 1

print(f"Board:\n{board}\n")
print(f"Pi:\n{pi}")

symmetries = g.getSymmetries(board, pi.flatten())
boards_array = np.array([b for b, _ in symmetries])

u, c = np.unique(boards_array, axis=0, return_counts=True)
print(f"Unique symmetries: {len(u)}")

for b, p in symmetries:
    p = np.array(p).reshape(2, 4, 4)
    print(f"Board:\n{b}\n")
    # print(f"Pi:\n{p}\n")
