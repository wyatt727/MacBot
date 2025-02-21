import os

def print_board(board):
    for row in board:
        print(" | ".join(row))
        print("-" * 5)

def check_winner(board, player):
    # Check rows
    for row in board:
        if all([cell == player for cell in row]):
            return True
    # Check columns
    for col in range(3):
        if all([board[row][col] == player for row in range(3)]):
            return True
    # Check diagonals
    if all([board[i][i] == player for i in range(3)]) or all([board[i][2 - i] == player for i in range(3)]):
        return True
    return False

def get_free_positions(board):
    free_positions = []
    for row in range(3):
        for col in range(3):
            if board[row][col] == " ":
                free_positions.append((row, col))
    return free_positions

def make_move(board, position, player):
    board[position[0]][position[1]] = player

def tic_tac_toe():
    board = [[" " for _ in range(3)] for _ in range(3)]
    current_player = "X"
    
    while True:
        print_board(board)
        free_positions = get_free_positions(board)
        
        if not free_positions:
            print("It's a tie!")
            break
        
        # Get move from current player
        row, col = map(int, input(f"Player {current_player}, enter your move (row and column): ").split())
        
        while (row, col) not in free_positions:
            print("Invalid move. Try again.")
            row, col = map(int, input(f"Player {current_player}, enter your move (row and column): ").split())
        
        make_move(board, (row, col), current_player)
        
        # Check for winner
        if check_winner(board, current_player):
            print_board(board)
            print(f"Player {current_player} wins!")
            break
        
        # Switch player
        current_player = "O" if current_player == "X" else "X"

tic_tac_toe()
