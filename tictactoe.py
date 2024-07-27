import math

def print_board(board):
    for row in board:
        print(" | ".join(row))
        print("-" * 5)

def check_winner(board):
    for row in board:
        if row[0] == row[1] == row[2] and row[0] != ' ':
            return row[0]
    for col in range(3):
        if board[0][col] == board[1][col] == board[2][col] and board[0][col] != ' ':
            return board[0][col]
    if board[0][0] == board[1][1] == board[2][2] and board[0][0] != ' ':
        return board[0][0]
    if board[0][2] == board[1][1] == board[2][0] and board[0][2] != ' ':
        return board[0][2]
    return None

def is_moves_left(board):
    for row in board:
        if ' ' in row:
            return True
    return False

def minimax(board, depth, is_max):
    winner = check_winner(board)
    if winner == 'X':
        return -10 + depth
    if winner == 'O':
        return 10 - depth
    if not is_moves_left(board):
        return 0
    
    if is_max:
        best = -math.inf
        for i in range(3):
            for j in range(3):
                if board[i][j] == ' ':
                    board[i][j] = 'O'
                    best = max(best, minimax(board, depth + 1, not is_max))
                    board[i][j] = ' '
        return best
    else:
        best = math.inf
        for i in range(3):
            for j in range(3):
                if board[i][j] == ' ':
                    board[i][j] = 'X'
                    best = min(best, minimax(board, depth + 1, not is_max))
                    board[i][j] = ' '
        return best

def find_best_move(board):
    best_val = -math.inf
    best_move = (-1, -1)
    for i in range(3):
        for j in range(3):
            if board[i][j] == ' ':
                board[i][j] = 'O'
                move_val = minimax(board, 0, False)
                board[i][j] = ' '
                if move_val > best_val:
                    best_move = (i, j)
                    best_val = move_val
    return best_move

def tic_tac_toe():
    board = [[' ' for _ in range(3)] for _ in range(3)]
    human_turn = True
    
    while True:
        print_board(board)
        if check_winner(board) or not is_moves_left(board):
            break
        if human_turn:
            print("Your move (enter row and column numbers 1-3): ")
            row, col = map(int, input().split())
            if board[row-1][col-1] == ' ':
                board[row-1][col-1] = 'X'
                human_turn = False
            else:
                print("Invalid move, try again.")
        else:
            print("AI's move:")
            row, col = find_best_move(board)
            board[row][col] = 'O'
            human_turn = True

    print_board(board)
    winner = check_winner(board)
    if winner:
        print(f"The winner is {winner}!")
    else:
        print("It's a tie!")

# Run the Tic-Tac-Toe game
tic_tac_toe()