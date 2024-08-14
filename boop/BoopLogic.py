from math import copysign
import numpy as np

class Board:
    directions = [(1,1),(1,0),(1,-1),(0,-1),(-1,-1),(-1,0),(-1,1),(0,1)]

    def __init__(self, n=6):
        self.n = n
        self.pieces = [[0 for _ in range(n)] for _ in range(n)]
        self.pieces.append([8, 0, 8, 0, 0, 0])
        # self.pieces = [[0, 0, 0, 2, 0, 0],
        #                [0, 2, 0, 0, 0, 0],
        #                [0, 0, 2, 0, 0, 0],
        #                [0, 2, 0, 0, 0, 0],
        #                [0, 0, 2, 0, 2, 0],
        #                [0, 0, 0, 0, 0, 2],
        #                [0, 1, 8, 0, 0, 0]]
        # self.pieces = [[0, 1, 0, 1, 0, 0],
        #                [0, 0, 0, 0, 0, 0],
        #                [0, 0, 1, 0, 0, 0],
        #                [0, 1, 0, 1, 0, 0],
        #                [0, 0, 0, 0, 0, 0],
        #                [0, 1, 0, 1, 0, 0],
        #                [1, 0, 8, 0, 0, 0]]

    def __getitem__(self, index):
        return self.pieces[index]

    def get_free_squares(self):
        legal_moves = []
        for i in range(self.n):
            for j in range(self.n):
                if self.pieces[i][j] == 0:
                    legal_moves.append((i, j))
        return legal_moves

    def has_legal_moves(self):
        return self.get_free_squares() != []

    def _boop_adjacent(self, cell, piece):
        i, j = cell
        for di, dj in self.directions:
            ni, nj = i + di, j + dj
            
            # Check if the adjacent cell is out of bounds or empty
            if ni < 0 or ni >= self.n or nj < 0 or nj >= self.n or self.pieces[ni][nj] == 0:
                continue
            
            adjacent_piece = self.pieces[ni][nj]
            
            # Kitten cannot boop a larger cat
            if abs(adjacent_piece) > abs(piece):
                continue

            next_i, next_j = i + 2 * di, j + 2 * dj
            
            # If the target cell to boop to is within bounds and empty, move the piece
            if 0 <= next_i < self.n and 0 <= next_j < self.n:
                if self.pieces[next_i][next_j] == 0:
                    self.pieces[next_i][next_j] = adjacent_piece
                    self.pieces[ni][nj] = 0
            else:
                # Otherwise, remove the piece and update the counts
                self._remove_piece((ni, nj), promote=False)

    
    def _remove_piece(self, cell, promote=True):
        i, j = cell
        piece = self.pieces[i][j]
        player_idx = 0 if piece > 0 else 1
        new_piece = 1 if promote else 0
        self.pieces[-1][player_idx*2 + new_piece] += 1
        self.pieces[i][j] = 0

    def _check_three(self):
        for i in range(self.n):
            for j in range(self.n):
                if self.pieces[i][j] != 0:
                    current_piece = self.pieces[i][j]
                    allowed_pieces = [copysign(x, current_piece) for x in [1, 2]]

                    def check_and_remove(cells):
                        if all(self.pieces[x][y] in allowed_pieces for x, y in cells):
                            for cell in cells:
                                self._remove_piece(cell)


                    # Check horizontal
                    if j + 2 < self.n:
                        check_and_remove([(i, j), (i, j + 1), (i, j + 2)])
                    # Check vertical
                    if i + 2 < self.n:
                        check_and_remove([(i, j), (i + 1, j), (i + 2, j)])
                    # Check diagonal (top-left to bottom-right)
                    if i + 2 < self.n and j + 2 < self.n:
                        check_and_remove([(i, j), (i + 1, j + 1), (i + 2, j + 2)])
                    # Check diagonal (top-right to bottom-left)
                    if i + 2 < self.n and j - 2 >= 0:
                        check_and_remove([(i, j), (i + 1, j - 1), (i + 2, j - 2)])
    
    def is_win(self, player):
        """Check if any set of three cats are all the same color, including diagonals. Return True if found, else False."""
        player_sets = [0, 0]
        for i in range(self.n):
            for j in range(self.n):
                if abs(self.pieces[i][j]) == 2:
                    piece = self.pieces[i][j]
                    # Check horizontal
                    if j + 2 < self.n and self.pieces[i][j+1] == piece and self.pieces[i][j+2] == piece:
                        if piece > 0:
                            player_sets[0] += 1
                        else:
                            player_sets[1] += 1
                    # Check vertical
                    if i + 2 < self.n and self.pieces[i+1][j] == piece and self.pieces[i+2][j] == piece:
                        if piece > 0:
                            player_sets[0] += 1
                        else:
                            player_sets[1] += 1
                    # Check diagonal (top-left to bottom-right)
                    if i + 2 < self.n and j + 2 < self.n and self.pieces[i+1][j+1] == piece and self.pieces[i+2][j+2] == piece:
                        if piece > 0:
                            player_sets[0] += 1
                        else:
                            player_sets[1] += 1
                    # Check diagonal (top-right to bottom-left)
                    if i + 2 < self.n and j - 2 >= 0 and self.pieces[i+1][j-1] == piece and self.pieces[i+2][j-2] == piece:
                        if piece > 0:
                            player_sets[0] += 1
                        else:
                            player_sets[1] += 1
        
        player_idx = 0 if player == 1 else 1
        if player_sets[player_idx] == player_sets[1 - player_idx]:
            # Both players have the same number of sets of three cats
            if player_sets[player_idx] > 0:
                # Both players have at least one set of three cats: draw
                return 1e-4
            # check if players have placed all their cats
            count = np.count_nonzero(self.pieces == player * 2)
            if count == 8:
                return 1
            count_opponent = np.count_nonzero(self.pieces == -player * 2)
            if count_opponent == 8:
                return -1
            # game is not over
            return 0
        if player_sets[player_idx] > player_sets[1 - player_idx]:
            # Player has more sets of three cats than opponent
            return 1 
        return -1


    def execute_move(self, move, piece):
        """Perform the given move on the board;
        piece is +/-1 for kitten and +/-2 for cat.
        """
        i, j = move
        assert self.pieces[i][j] == 0
        self.pieces[i][j] = piece
        player = 0 if piece > 0 else 1
        self.pieces[-1][player * 2 + abs(piece) - 1] -= 1
        self._boop_adjacent((i, j), piece)
        if not self.is_win(copysign(1, piece)):
            self._check_three()