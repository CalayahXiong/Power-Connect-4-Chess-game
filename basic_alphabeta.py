import time

import numpy as np
import math

from matplotlib import pyplot as plt

ROW_COUNT = 8
COLUMN_COUNT = 8

EMPTY = " "

WHITE = 0
BLACK = 1

WINDOW_LENGTH = 4

visited_states = 0


def create_board():
    board = np.full((ROW_COUNT, COLUMN_COUNT), EMPTY)
    return board


def drop_piece(board, row, col, piece):
    board[row][col] = piece


def is_valid_location(board, col):
    return board[ROW_COUNT - 1][col] == EMPTY


def get_next_open_row(board, col):
    for r in range(ROW_COUNT):
        if board[r][col] == EMPTY:
            return r


def print_board(board):
    print(np.flip(board, 0))


def winning_move(board, piece):
    # Check horizontal locations for win
    for c in range(COLUMN_COUNT - 3):
        for r in range(ROW_COUNT):
            if (-1 if board[r][c] == ' ' else int(board[r][c])) == piece \
                    and (-1 if board[r][c+1] == ' ' else int(board[r][c+1])) == piece \
                    and (-1 if board[r][c+2] == ' ' else int(board[r][c+2])) == piece \
                    and (-1 if board[r][c+3] == ' ' else int(board[r][c+3])) == piece:
                return True

    # Check vertical locations for win
    for c in range(COLUMN_COUNT):
        for r in range(ROW_COUNT - 3):
            if (-1 if board[r][c] == ' ' else int(board[r][c])) == piece \
                    and (-1 if board[r+1][c] == ' ' else int(board[r+1][c])) == piece \
                    and (-1 if board[r+2][c] == ' ' else int(board[r+2][c])) == piece \
                    and (-1 if board[r+3][c] == ' ' else int(board[r+3][c])) == piece:
                return True

    # Check positively sloped diagonals
    for c in range(COLUMN_COUNT - 3):
        for r in range(ROW_COUNT - 3):
            if (-1 if board[r][c] == ' ' else int(board[r][c])) == piece \
                    and (-1 if board[r+1][c+1] == ' ' else int(board[r+1][c+1])) == piece \
                    and (-1 if board[r+2][c+2] == ' ' else int(board[r+2][c+2])) == piece \
                    and (-1 if board[r+3][c+3] == ' ' else int(board[r+3][c+3])) == piece:
                return True

    # Check negatively sloped diagonals
    for c in range(COLUMN_COUNT - 3):
        for r in range(3, ROW_COUNT):
            if (-1 if board[r][c] == ' ' else int(board[r][c])) == piece and (
            -1 if board[r - 1][c - 1] == ' ' else int(board[r - 1][c - 1])) == piece \
                    and (-1 if board[r - 2][c - 2] == ' ' else int(board[r - 2][c - 2])) == piece and (
            -1 if board[r - 3][c - 3] == ' ' else int(board[r - 3][c - 3])) == piece:
                return True


def get_valid_locations(board):
    valid_locations = []
    for col in range(COLUMN_COUNT):
        if is_valid_location(board, col):
            valid_locations.append(col)
    return valid_locations


def count_runs(board, piece):
    count = 0
    run_length = 0

    # Check rows
    for row in range(len(board)):
        for col in range(len(board[0])):
            if board[row][col] == piece:
                run_length += 1
            else:
                if run_length >= 2:
                    count += 1
                run_length = 0
        if run_length >= 2:  # Check at end of row
            count += 1
        run_length = 0  # Reset for the next row

    # Check columns
    for col in range(len(board[0])):
        for row in range(len(board)):
            if board[row][col] == piece:
                run_length += 1
            else:
                if run_length >= 2:
                    count += 1
                run_length = 0
        if run_length >= 2:  # Check at end of column
            count += 1
        run_length = 0  # Reset for the next column

    # Check diagonals (bottom-left to top-right)
    for r in range(len(board)):
        for c in range(len(board[0])):
            if r + 1 < len(board) and c + 1 < len(board[0]):
                if board[r][c] == piece and board[r + 1][c + 1] == piece:
                    run_length += 1
                    if run_length == 2:  # Increment when we get the second piece in a run
                        count += 1
                else:
                    run_length = 0

    # Check diagonals (top-left to bottom-right)
    for r in range(len(board)):
        for c in range(len(board[0])):
            if r - 1 >= 0 and c + 1 < len(board[0]):
                if board[r][c] == piece and board[r - 1][c + 1] == piece:
                    run_length += 1
                    if run_length == 2:  # Increment when we get the second piece in a run
                        count += 1
                else:
                    run_length = 0

    return count


def basic_evaluation(board, playerPiece):
    # Count runs for player and opponent
    player_runs = count_runs(board, playerPiece)
    opponentPiece = 1 if playerPiece == 0 else 0
    opponent_runs = count_runs(board, opponentPiece)

    # Calculate H(n) = Runs for player - Runs for opponent
    return player_runs - opponent_runs


def minimax(board, depth, alpha, beta, maximizingPlayer, playerPiece):
    global visited_states
    visited_states += 1

    valid_locations = get_valid_locations(board)

    if len(valid_locations) == 0:  # board is full
        return (None, None, "full")

    is_terminal = is_terminal_node(board, playerPiece)

    opponentPiece = 1 if playerPiece == 0 else 0

    if depth == 0 or is_terminal:
        if is_terminal:
            if winning_move(board, playerPiece):
                return (None, 10000000000000, "")  # Player wins
            elif winning_move(board, 1 if playerPiece == 0 else 0):
                return (None, -100000000000000, "")  # Opponent wins
            else:  # Game is over, no more valid moves
                return (None, 0, "")
        else:  # Depth is zero
            return (None, basic_evaluation(board, playerPiece), "")


    if maximizingPlayer:
        value = -math.inf
        best_move = None
        best_instruction = ""

        # Evaluate sliding moves (L, R)
        for row in range(ROW_COUNT):
            for col in range(COLUMN_COUNT):
                if board[row][col] == opponentPiece:
                    # Check left slide
                    if can_slide(board, opponentPiece, row, col, 'L', 3):
                        b_copy = board.copy()
                        slide_pieces(b_copy, row, col, 'L', 3)
                        new_score = minimax(b_copy, depth - 1, alpha, beta, False, opponentPiece)[1]
                        if new_score > value:
                            value = new_score
                            best_instruction = f"L {col + 1} {row + 1}"

                        # Check for pruning
                        alpha = max(alpha, value)
                        if alpha >= beta:
                            return best_move, value, best_instruction  # prune

                    # Check right slide
                    if can_slide(board, opponentPiece, row, col, 'R', 3):
                        b_copy = board.copy()
                        slide_pieces(b_copy, row, col, 'R', 3)
                        new_score = minimax(b_copy, depth - 1, alpha, beta, False, opponentPiece)[1]
                        if new_score > value:
                            value = new_score
                            best_instruction = f"R {col + 1} {row + 1}"

                        # Check for pruning
                        alpha = max(alpha, value)
                        if alpha >= beta:
                            return best_move, value, best_instruction  # prune

        # Evaluate drop moves (D)
        for col in valid_locations:
            row = get_next_open_row(board, col)
            b_copy = board.copy()
            drop_piece(b_copy, row, col, playerPiece)
            new_score = minimax(b_copy, depth - 1, alpha, beta, False, opponentPiece)[1]
            if new_score > value:
                value = new_score
                best_move = col
                best_instruction = f"D {col + 1}"  # Column for drop

            alpha = max(alpha, value)
            if alpha >= beta:
                break  # prune

        return best_move, value, best_instruction

    else:
        value = math.inf
        best_move = None
        best_instruction = ""

        # Evaluate sliding moves (L, R) for minimizing player
        for row in range(ROW_COUNT):
            for col in range(COLUMN_COUNT):
                if board[row][col] == playerPiece:
                    # Check left slide
                    if can_slide(board, playerPiece, row, col, 'L', 3):
                        b_copy = board.copy()
                        slide_pieces(b_copy, row, col, 'L', 3)
                        new_score = minimax(b_copy, depth - 1, alpha, beta, True, playerPiece)[1]
                        if new_score < value:
                            value = new_score
                            best_instruction = f"L {col + 1} {row + 1}"

                        # Check for pruning
                        beta = min(beta, value)
                        if alpha >= beta:
                            return best_move, value, best_instruction  # prune

                    # Check right slide
                    if can_slide(board, playerPiece, row, col, 'R', 3):
                        b_copy = board.copy()
                        slide_pieces(b_copy, row, col, 'R', 3)
                        new_score = minimax(b_copy, depth - 1, alpha, beta, True, playerPiece)[1]
                        if new_score < value:
                            value = new_score
                            best_instruction = f"R {col + 1} {row + 1}"

                        # Check for pruning
                        beta = min(beta, value)
                        if alpha >= beta:
                            return best_move, value, best_instruction  # prune

        # Evaluate drop moves (D)
        for col in valid_locations:
            row = get_next_open_row(board, col)
            b_copy = board.copy()
            drop_piece(b_copy, row, col, playerPiece)
            new_score = minimax(b_copy, depth - 1, alpha, beta, True, playerPiece)[1]
            if new_score < value:
                value = new_score
                best_move = col
                best_instruction = f"D {col + 1}"

            beta = min(beta, value)
            if alpha >= beta:
                break  # prune

        return best_move, value, best_instruction



def measure_visited_states_for_depths(board, myAgent, max_depths):
    visited_states_for_depths = []
    for depth in max_depths:
        global visited_states
        visited_states = 0
        minimax(board, depth, -math.inf, math.inf, True, myAgent)
        visited_states_for_depths.append(visited_states)
    return visited_states_for_depths


def plot_visited_states(max_depths, visited_states_for_depths):
    plt.plot(max_depths, visited_states_for_depths, marker='o')
    plt.title("States Visited (Basic_Alphabeta)")
    plt.xlabel("Depth Cutoff")
    plt.ylabel("Total Visited States")
    plt.grid(True)
    plt.show()



def is_terminal_node(board, playerPiece):
    opponentPiece = 1 if playerPiece == 0 else 0
    return winning_move(board, playerPiece) or winning_move(board, opponentPiece) or len(
        get_valid_locations(board)) == 0


def can_slide(board, myPiece, row, col, direction, slideLength):
    if (row > ROW_COUNT - 1 or col > COLUMN_COUNT - 1 or board[row][col] == ' ' or board[row][col] != myPiece):
        return False  # No piece to slide or a wrong place be chosen

    opponent = '1' if myPiece == '0' else '0'

    if (direction == 'L' and col >= slideLength):
        # check the sequential pieces
        for i in range(slideLength):
            if (board[row][col - i] != myPiece):
                return False
        if (board[row][col - slideLength] == opponent):
            return (col - slideLength - 1 < 0 or board[row][col - slideLength - 1] == ' ')

    elif (direction == 'R' and col <= COLUMN_COUNT - 1 - slideLength):
        for i in range(slideLength):
            if (board[row][col + i] != myPiece):
                return False
        if (board[row][col + slideLength] == opponent):
            return (col + slideLength + 1 > COLUMN_COUNT or board[row][col + slideLength + 1] == ' ')


def slide_pieces(board, row, col, direction, slideLength):
    # Shift pieces left or right
    if direction == "L":
        # Move left
        # horizontal
        if col - slideLength - 1 > 0:
            board[row][col - slideLength - 1] = board[row][col - slideLength]
        for i in range(0, slideLength):
            board[row][col - slideLength + i] = board[row][col - slideLength + i + 1]
        # vertical
        for r in range(0, row):
            board[row - r][col] = board[row - r - 1][col]

    elif direction == "R":
        # Move right
        # horizontal
        if col + slideLength + 1 < COLUMN_COUNT:
            board[row][col + slideLength + 1] = board[row][col + slideLength]
        for i in range(0, slideLength):
            board[row][col + slideLength - i] = board[row][col + slideLength - i - 1]
        # vertical
        for r in range(0, row):
            board[row - r][col] = board[row - r - 1][col]


def measure_time_for_depths(board, myAgent, max_depths):
    times_for_depths = []
    opponent = 1 if myAgent == 0 else 0
    valid_locations = get_valid_locations(board)

    for depth in max_depths:
        total_time = 0
        num_moves = 3  # We will measure for 3 opening moves

        for _ in range(num_moves):
            start_time = time.time()
            minimax(board, depth, -math.inf, math.inf, True, myAgent)
            end_time = time.time()
            total_time += end_time - start_time

        avg_time = total_time / num_moves
        times_for_depths.append(avg_time)

    return times_for_depths


def plot_times(max_depths, times_for_depths):
    plt.plot(max_depths, times_for_depths, marker='o')
    plt.title("Basic_Alphabeta")
    plt.xlabel("Depth Cutoff")
    plt.ylabel("Average Time (seconds)")
    plt.grid(True)
    plt.show()


def play_game(myAgent, depth):
    board = create_board()
    print_board(board)
    game_over = False

    turn = WHITE  # White starts first
    opponent = 1 if myAgent == 0 else 0

    while not game_over:
        # This can be replaced by player from other CP
        if turn == opponent:
            # Read instruction from opponent
            step = input("Enter your move (D <x>, L <x y>, R <x y>):")
            parts = step.split()

            if parts[0] == 'D':  # Drop piece
                col = int(parts[1]) - 1
                if is_valid_location(board, col):
                    row = get_next_open_row(board, col)
                    drop_piece(board, row, col, opponent)

            elif parts[0] in ['L', 'R']:  # Slide pieces
                direction = parts[0]
                x = int(parts[1]) - 1
                y = int(parts[2]) - 1

                if can_slide(board, opponent, y, x, direction, 3):
                    slide_pieces(board, y, x, direction, 3)
                elif can_slide(board, opponent, y, x, direction, 2):
                    slide_pieces(board, y, x, direction, 2)

            print_board(board)

            if winning_move(board, opponent):
                print(f"Player {opponent} wins!!")
                game_over = True

            turn = (turn + 1) % 2  # Switch turn


        if turn == myAgent:
            print("my agent is making decision...")
            col, minimax_score, instruction = minimax(board, depth, -math.inf, math.inf, True, myAgent)
            print(f"My agent decided: {instruction}")

            # Execute the chosen action
            if instruction.startswith("D"):
                col = int(instruction.split()[1]) - 1
                row = get_next_open_row(board, col)
                drop_piece(board, row, col, myAgent)
            elif instruction.startswith("L") or instruction.startswith("R"):
                direction = instruction[0]
                x = int(instruction.split()[1]) - 1
                y = int(instruction.split()[2]) - 1
                if can_slide(board, myAgent, y, x, direction, 3):
                    slide_pieces(board, y, x, direction, 3)

            if winning_move(board, myAgent):
                print(f"Player {myAgent} wins!!")
                game_over = True

            print_board(board)
            turn = (turn + 1) % 2


# Which piece you wann your agent to be
# play_game(WHITE, 5)
#
# Plot part
# Initialize the board and agent settings
board = create_board()
myAgent = 0  # agent plays WHITE piece

# Define the depth cutoffs we want to test
max_depths = [1, 2, 3, 4, 5, 6]

# Measure the times for different depths
times_for_depths = measure_time_for_depths(board, myAgent, max_depths)

#Plot the graph
plot_times(max_depths, times_for_depths)

# Plot states
# visited_states_for_depths = measure_visited_states_for_depths(board, myAgent, max_depths)
# plot_visited_states(max_depths, visited_states_for_depths)