# This code was generated by chatGPT
import sys

def is_safe(board, row, col, N):
    # check if there is a queen in the same column
    for i in range(row):
        if board[i][col] == 1:
            return False

    # check if there is a queen in the top-left diagonal
    i = row
    j = col
    while i >= 0 and j >= 0:
        if board[i][j] == 1:
            return False
        i -= 1
        j -= 1

    # check if there is a queen in the top-right diagonal
    i = row
    j = col
    while i >= 0 and j < N:
        if board[i][j] == 1:
            return False
        i -= 1
        j += 1

    return True

def solve_n_queens(board, row, N):
    if row == N:
        # we have placed all the queens, so we print the solution
        for i in range(N):
            print(" ".join([str(cell) for cell in board[i]]))
        return True

    # try placing a queen in the current row and check if it is safe
    for col in range(N):
        if is_safe(board, row, col, N):
            # place the queen and move on to the next row
            board[row][col] = 1
            if solve_n_queens(board, row + 1, N):
                return True
            # backtrack
            board[row][col] = 0

    return False

def n_queens(N):
    # create an empty board
    board = [[0] * N for _ in range(N)]
    if not solve_n_queens(board, 0, N):
        print("No solution")
def main():
	if len(sys.argv) != 2:
		print("Usage: nqueens N")
		sys.exit(1)

	try:
		N = int(sys.argv[1])
	except ValueError:
		print("N must be a number")
		sys.exit(1)

	if N < 4:
		print("N must be at least 4")
		sys.exit(1)

	# test the function
	n_queens(N)
if __name__ == "__main__":
    main()