from code.Algorithms import DFS, BFS, Astar, RandomMove, RandomLegalMove, RandomLegalRepeatMove
from code.Classes import GameBoard, GameFile, History
from code.Visualisation import visualise
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import pygame
import timeit
import csv


def load_board_opstellingen(path: str) -> list[str]:
    '''
    Zet alle gegeven bord csv's in een folder
    geef van deze folder de path als input
    krijg een mooie lijst me paths naar elke bord file terug
    input:
    - path: string = het relative path voor de folder waarin alle csv files voor de borden in staan

    output:
    - return: list[string] = voor elke file in de aangewezen folder -> relatieve pad + filenaam
    '''
    return [os.path.join(path, file) for file in os.listdir(path)]


def pick_board_random() -> str:
    '''
    Kiest een random board
    '''
    return random.choice(load_board_opstellingen('data'))


def available_boards():
    print("\nAvailable boards:\n")
    # Hier staan alle borden.
    boards_dictionary = {
        '1': "data/Rushhour6x6_1.csv",
        '2': "data/Rushhour6x6_2.csv",
        '3': "data/Rushhour6x6_3.csv",
        '4': "data/Rushhour9x9_4.csv",
        '5': "data/Rushhour9x9_5.csv",
        '6': "data/Rushhour9x9_6.csv",
        '7': "data/Rushhour12x12_7.csv",
    }

    board_sizes = ['6x6', '6x6', '6x6', '9x9', '9x9', '9x9', '12x12']

    for i in range(len(boards_dictionary)):
        print(f'{i + 1}: {board_sizes[i]}\n')

    return boards_dictionary


def available_algorithms():
    print("\nAvailable algorithms:\n")
    # Hierin kun je de beschikbare algoritmes plaatsen!
    algorithms_dictionary = {
        "1": RandomLegalMove,
        "2": RandomLegalRepeatMove,
        "3": RandomMove,
    }

    algorithm_names = ['RandomLegalMove', 'RandomLegalRepeatMove', 'RandomMove']

    for i in range(len(algorithms_dictionary)):
        print(f'{i + 1}: {algorithm_names[i]}\n')

    return algorithms_dictionary


def visualize_random():
    """
    Main function to run the Rush Hour game.
    """
    # Create an instance of the History class
    history = History()

    if input("Random board? yes/no : ") == 'yes':
        file_path = pick_board_random()
    else:
        available_board_dictionary = available_boards()
        board_pick = str
        while board_pick not in available_board_dictionary:
            board_pick = str(input("Which board will you pick? "))

        file_path = available_board_dictionary[board_pick]

    algorithms = available_algorithms()
    select_algorithm = str
    while select_algorithm not in algorithms:
        select_algorithm = input("Choose an algorithm: ").lower()

        selected_algorithm = algorithms[select_algorithm]

    game_file = GameFile(file_path)
    game = GameBoard(game_file)

    while True:

        if game.is_won():
            iterative_gameplay(history.get_move_history(), file_path)
            print("Congratulations, you found your way out!")
            print('Total moves:', history.get_counter())
            break

        random_move_algorithm = selected_algorithm(game, history, game_file)
        random_car, random_direction = random_move_algorithm.make_move()

        if game.move_car(random_car, random_direction):
            history.add_move(random_car, random_direction)
            history.add_board(game.get_board())


def experiment():
    """
    Used for comparing the random algorithms, runs the rushhour game as many times as the input given in main().
    """

    # needed moves word hierin opgeslagen na solve
    total_moves = []

    # illegal moves worden niet bijgehouden, dus tel OOK hoevaak de game loop gerund word
    total_loops = []

    number_of_games = int(input("\nHow many games do you want to run for this experiment? "))

    available_board_dictionary = available_boards()
    board_pick = str
    while board_pick not in available_board_dictionary:
        board_pick = str(input("Which board will you pick? "))

    algorithms = available_algorithms()
    select_algorithm = str
    while select_algorithm not in algorithms:
        select_algorithm = input("Choose an algorithm: ").lower()

    if select_algorithm in ['1', '2', '3']:
        selected_algorithm = algorithms[select_algorithm]

    for i in range(number_of_games):

        history = History()

        file_path = available_board_dictionary[board_pick]

        game_file = GameFile(file_path)
        game = GameBoard(game_file)

        loop_counter = 0

        while True:

            if game.is_won():
                print(f"game {i + 1} was solved in {history.get_counter()} moves, and {loop_counter} game loops")
                total_moves.append(history.get_counter())
                total_loops.append(loop_counter)
                break

            random_move_algorithm = selected_algorithm(game, history, game_file)
            random_car, random_direction = random_move_algorithm.make_move()

            if game.move_car(random_car, random_direction):
                history.add_move(random_car, random_direction)
                history.add_board(game.get_board())

            loop_counter += 1

    average_moves = (sum(total_moves) / len(total_moves))
    average_loops = (sum(total_loops) / len(total_loops))

    print(f"\nThe average amount of moves needed for {number_of_games} games was {average_moves} moves, and {average_loops} game loops")


def breadth_first_search(board_path=None):

    # If board_path is not provided, prompt the user
    if board_path is None:
        available_board_dictionary = available_boards()
        board_pick = str
        while board_pick not in available_board_dictionary:
            board_pick = str(input("Which board will you pick? "))
        file_path = available_board_dictionary[board_pick]
    else:
        file_path = board_path

    game_file = GameFile(file_path)
    game = GameBoard(game_file)

    game.get_board_for_player()

    # start timer for data here
    start_time = timeit.default_timer()

    bfs_instance = BFS(game)
    bfs_result = bfs_instance.run()

    # end timer for data here
    end_time = timeit.default_timer()
    compute_time = end_time - start_time

    if bfs_result is not None:
        solution_path, visited_states_count = bfs_result
        iterative_gameplay(solution_path, file_path)
        print(f"results: \n solution_path = {len(solution_path)}\n visited_states_count: {visited_states_count}")

        # Export the compressed BFS move history to a CSV file
        bfs_instance.csv_output()

        # Store data in a CSV file
        output_file_path = "data/algoritmen_data.csv"
        with open(output_file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["BFS", file_path, compute_time, len(solution_path), visited_states_count])

        print(f"Results saved in {output_file_path}")
    else:
        print("BFS did not find a solution.")


def depth_first_search(board_path=None):
    # If board_path is not provided, prompt the user
    if board_path is None:
        available_board_dictionary = available_boards()
        board_pick = str
        while board_pick not in available_board_dictionary:
            board_pick = str(input("Which board will you pick? "))
        file_path = available_board_dictionary[board_pick]
    else:
        file_path = board_path

    game_file = GameFile(file_path)
    game = GameBoard(game_file)

    game.get_board_for_player()

    # start timer for data here
    start_time = timeit.default_timer()

    dfs_instance = DFS(game)
    dfs_result = dfs_instance.run()

    # end timer for data here
    end_time = timeit.default_timer()
    compute_time = end_time - start_time

    if dfs_result is not None:
        solution_path, visited_states_count = dfs_result
        iterative_gameplay(solution_path, file_path)
        print(f"results: \n solution_path = {len(solution_path)}\n visited_states_count: {visited_states_count}")

        # Export the compressed DFS move history to a CSV file
        dfs_instance.csv_output()

        # Store data in a CSV file
        output_file_path = "data/algoritmen_data.csv"
        with open(output_file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["DFS", file_path, compute_time, len(solution_path), visited_states_count])

        print(f"Results saved in {output_file_path}")
    else:
        print("DFS did not find a solution.")


def astar_algorithm(board_path=None):
    # If board_path is not provided, prompt the user
    if board_path is None:
        available_board_dictionary = available_boards()
        board_pick = str
        while board_pick not in available_board_dictionary:
            board_pick = str(input("Which board will you pick? "))
        file_path = available_board_dictionary[board_pick]
    else:
        file_path = board_path

    game_file = GameFile(file_path)
    game = GameBoard(game_file)

    game.get_board_for_player()

    # start timer for data here
    start_time = timeit.default_timer()

    astar_instance = Astar(game)
    astar_result = astar_instance.run()

    # end timer for data here
    end_time = timeit.default_timer()
    compute_time = end_time - start_time

    if astar_result is not None:
        solution_path, visited_states_count = astar_result
        iterative_gameplay(solution_path, file_path)
        print(f"results: \n solution_path = {len(solution_path)}\n visited_states_count: {visited_states_count}")

        # Export the compressed Astar move history to a CSV file
        astar_instance.csv_output()

        # Store data in a CSV file
        output_file_path = "data/algoritmen_data.csv"
        with open(output_file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            # writer.writerow(["Algorithm", "Board", "Compute Time", "Solution Path Length", "Visited States Count"])
            writer.writerow(["ASTAR", file_path, compute_time, len(solution_path), visited_states_count])

        print(f"Results saved in {output_file_path}")
    else:
        print("Astar did not find a solution.")


def compare_BFS_DFS_ASTAR(csv_data_file):
    """
    Reads the data file made by running algo_comparisons() or "auto" mode functions
    Makes a barchart from the data in your algoritmen_data.csv file
    """

    Algorithms = []
    Board = []
    Compute_Time = []
    Solution_Path_Length = []
    Visited_States_Count = []

    # Open the CSV file for reading
    with open(csv_data_file, mode='r') as file:
        reader = csv.reader(file)
        # Skip the header row
        next(reader)

        # Read data from each row and append to respective lists
        for row in reader:
            Algorithms.append(row[0])
            Board.append(row[1])
            Compute_Time.append(float(row[2]))
            Solution_Path_Length.append(int(row[3]))
            Visited_States_Count.append(int(row[4]))
    
    # The forloops were generated with chatgpt because of its complexity
    # Identify unique algorithms and boards
    unique_algorithms = sorted(list(set(Algorithms)))
    unique_boards = sorted(list(set(Board)))

    # Create grouped bar chart for Solution Path Length
    x_pos_solution = np.arange(len(unique_boards))
    width = 0.25  # the width of the bars

    plt.figure(figsize=(12, 6))  # Set the figure size

    # Create subplot for Solution Path Length
    plt.subplot(1, 2, 1)

    for i, algorithm in enumerate(unique_algorithms):
        algorithm_indices = [j for j, alg in enumerate(Algorithms) if alg == algorithm]
        values = [Solution_Path_Length[idx] for idx in algorithm_indices]

        plt.bar(x_pos_solution + i * width, values, width, label=algorithm)

        # Add numbers above the bars
        for j, value in enumerate(values):
            plt.text(x_pos_solution[j] + i * width, value + 0.1, str(value), ha='center', va='bottom')

    plt.xlabel("Boards")
    plt.ylabel("Solution Path Length")
    plt.title("Comparison of Solution Path Length on 6x6 boards")
    plt.xticks(x_pos_solution + (len(unique_algorithms) - 1) * width / 2, unique_boards, rotation=45)  # Adjust xticks position and rotation
    plt.legend()

    # Create subplot for Visited States Count
    plt.subplot(1, 2, 2)

    x_pos_visited_states = np.arange(len(unique_boards))

    for i, algorithm in enumerate(unique_algorithms):
        algorithm_indices = [j for j, alg in enumerate(Algorithms) if alg == algorithm]
        values = [Visited_States_Count[idx] for idx in algorithm_indices]

        plt.bar(x_pos_visited_states + i * width, values, width, label=algorithm)

        # Add numbers above the bars with an offset for the second set of bars
        for j, value in enumerate(values):
            plt.text(x_pos_visited_states[j] + i * width, value + 0.1, str(value), ha='center', va='bottom')

    plt.xlabel("Boards")
    plt.ylabel("Visited States Count")
    plt.title("Comparison of Visited States Count on 6x6 boards")
    plt.xticks(x_pos_visited_states + (len(unique_algorithms) - 1) * width / 2, unique_boards, rotation=45)  # Adjust xticks position and rotation
    plt.legend()

    # Save the plot to a file
    picture_path = str(input("Picture name? "))
    plt.savefig(picture_path)
    # Display a message to the user
    print(f"The result is saved as {picture_path}. Please check the files!")



def manual_algo_comparisons():
    """
    Compare the algorithms on each board manually
    """
    # Get the initial number of rows in the CSV file
    csv_data_file = 'data/algoritmen_data.csv'
    initial_row_count = get_csv_row_count(csv_data_file)

    while True:
        mode = str
        while mode not in ['b', 'd', 'q', 'a']:
            mode = input("\nbfs or dfs or astar or quit? (b/d/a/q) ").lower()
        if mode == 'b':
            breadth_first_search()
        elif mode == 'd':
            depth_first_search()
        elif mode == 'a':
            astar_algorithm()
        elif mode == 'q':
            break

        continu = str
        while continu not in ['q', 'c']:
            continu = input("\nDo you want to continue, or quit? (c/q) ")

        if continu == 'q':
            # Print only the rows added after running the tests
            new_row_count = get_csv_row_count(csv_data_file)
            print_added_rows(csv_data_file, initial_row_count, new_row_count)
            break
        elif continu == 'c':
            continue


def get_csv_row_count(csv_file):
    """
    looks at the algoritmen_data.csv file row count
    """
    with open(csv_file, mode='r') as file:
        reader = csv.reader(file)
        return sum(1 for _ in reader)


def print_added_rows(csv_file, initial_row_count, new_row_count):
    """
    prints the newest rows added to the algoritmen_data.csv when using algo mode
    """
    with open(csv_file, mode='r') as file:
        reader = csv.reader(file)
        rows = list(reader)[initial_row_count:new_row_count]

        header = ["Algorithm", "Board", "Compute Time", "Solution Path Length", "Visited States Count"]
        for row in rows:
            print("\nNew Test Results:")
            for key, value in zip(header, row):
                print(f"{key}: {value}")


def run_algorithms_on_boards(board_paths):
    """
    run the boards automatically on their board-length size
    """
    for board_path in board_paths:
        print(f"\nRunning algorithms on board: {board_path}")

        # BFS
        print("Running BFS:")
        breadth_first_search(board_path)

        # DFS
        print("Running DFS:")
        depth_first_search(board_path)

        # Astar
        print("Running Astar:")
        astar_algorithm(board_path)


def iterative_gameplay(paths: list[tuple[int, int]], file: str) -> None:
    """
    This function simulates gameplay on a Rush-Hour board using provided move paths.

    Pre: A path, including all the moves, is given, alongside game_file for simulation.
    Post: A pygame, displaying all the moves, is shown.
    """
    game_file = GameFile(file)
    visual = GameBoard(game_file)

    pygame.init()
    # The screen is established with the size of the rows and colums
    rows = len(visual._board)
    cols = len(visual._board[0])
    screen = pygame.display.set_mode((cols * 50, rows * 50))
    pygame.display.set_caption("Rush-Hour Board")
    clock = pygame.time.Clock()

    for move in paths:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        # For every move in the given path, the board is updated.
        visual.move_car(move[0], move[1])
        screen.fill((127, 127, 127))
        visual.draw_board(screen)
        pygame.display.flip()

        clock.tick(15)

    pygame.quit()


def main():

    while True:
        mode = str
        while mode not in ['v', 'e', 'algo', 'auto', 'p']:
            mode = input("\nDo you want to run the game in the Visualize-random mode, Experiment mode, manual_algo_comparison,\
OR automatical_algo_comparison OR print results from automatical_algo_comparison? (v/e/algo/auto/p) ").lower()
        if mode == 'v':
            visualize_random()
        elif mode == 'e':
            experiment()
        elif mode == 'algo':
            manual_algo_comparisons()
        elif mode == 'auto':
            board_size = int(input("Enter the board size (6, 9, or 12): "))
            if board_size == 6:
                board_paths = ["data/Rushhour6x6_1.csv", "data/Rushhour6x6_2.csv", "data/Rushhour6x6_3.csv"]
            elif board_size == 9:
                board_paths = ["data/Rushhour9x9_4.csv", "data/Rushhour9x9_5.csv", "data/Rushhour9x9_6.csv"]
            elif board_size == 12:
                board_paths = ["data/Rushhour12x12_7.csv"]
            else:
                print("Invalid board size. Please enter 6, 9, or 12.")
                continue

            run_algorithms_on_boards(board_paths)
        elif mode == 'p':
            csv_data_file = 'data/algoritmen_data.csv'
            compare_BFS_DFS_ASTAR(csv_data_file)
            break

        continu = str
        while continu not in ['q', 'c']:
            continu = input("\nDo you want to continue, or quit? (c/q) ")

        if continu == 'q':
            csv_data_file = 'data/algoritmen_data.csv'
            compare_BFS_DFS_ASTAR(csv_data_file)
            break
        elif continu == 'c':
            continue


if __name__ == '__main__':
    main()
