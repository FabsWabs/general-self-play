class Board:
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

    def __init__(self, n):
        self.n = n
        # self.pieces = [[0 for _ in range(n)] for _ in range(n)]
        self.pieces = [[1, 1, -1, 0], [1, -1, 1, 0], [1, 1, -1, 0], [0, 0, 0, 0]]

    def __getitem__(self, index):
        return self.pieces[index]

    def get_legal_moves(self):
        legal_moves = []
        for i in range(self.n):
            for j in range(self.n):
                if self.pieces[i][j] == 0:
                    legal_moves.append((i, j))
        return legal_moves

    def has_legal_moves(self):
        return any(self.pieces[i][j] == 0 for i in range(self.n) for j in range(self.n))

    def _flip_neighbors(self, i, j):
        """Flip all neighbors (excluding diagonals) of (i, j) from white to black or vice versa."""
        for di, dj in self.directions:
            ni, nj = i + di, j + dj
            if 0 <= ni < self.n and 0 <= nj < self.n and self.pieces[ni][nj] != 0:
                self.pieces[ni][nj] *= -1

    def is_win(self):
        """Check if any sets of n pieces are all the same color, including diagonals. Return the first set found or None."""
        # Check rows
        for i in range(self.n):
            if all(self.pieces[i][j] == 1 for j in range(self.n)) or all(
                self.pieces[i][j] == -1 for j in range(self.n)
            ):
                return True

        # Check columns
        for j in range(self.n):
            if all(self.pieces[i][j] == 1 for i in range(self.n)) or all(
                self.pieces[i][j] == -1 for i in range(self.n)
            ):
                return True

        # Check diagonals
        if all(self.pieces[i][i] == 1 for i in range(self.n)) or all(
            self.pieces[i][i] == -1 for i in range(self.n)
        ):
            return True

        if all(self.pieces[i][self.n - i - 1] == 1 for i in range(self.n)) or all(
            self.pieces[i][self.n - i - 1] == -1 for i in range(self.n)
        ):
            return True

        return False

    def execute_move(self, move, color):
        """Perform the given move on the board;
        color gives the color of the piece to play (1=white,-1=black)
        """
        i, j = move
        assert self.pieces[i][j] == 0
        self.pieces[i][j] = color
        self._flip_neighbors(i, j)
