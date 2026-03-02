import chess
import chess.pgn
import engine
import time
import datetime
import os

# Create a folder to store the generated training data
if not os.path.exists("training_data"):
    os.makedirs("training_data")

def play_self_match(game_number):
    board = chess.Board()
    pgn_game = chess.pgn.Game()
    
    # Metadata for the dataset
    pgn_game.headers["Event"] = f"Self-Play RL Generation #{game_number}"
    pgn_game.headers["Date"] = datetime.datetime.now().strftime("%Y.%m.%d")
    pgn_game.headers["White"] = "Neural Net V2"
    pgn_game.headers["Black"] = "Neural Net V2"
    
    node = pgn_game
    print(f"\n--- STARTING GAME {game_number} ---")

    while not board.is_game_over():
        start_time = time.time()
        
        # Adjust depth depending on how fast you want games generated
        # Depth 1 or 2 is fine for early reinforcement learning
        best_move = engine.get_best_move(board, depth=1) 
        
        if best_move is None:
            break

        board.push(best_move)
        node = node.add_variation(best_move)
        
        # Simple progress tracker
        if board.fullmove_number % 10 == 0 and board.turn == chess.BLACK:
             print(f"Move {board.fullmove_number} reached...")

    result = board.result()
    pgn_game.headers["Result"] = result
    print(f"Game {game_number} Over. Result: {result}")

    # Save to the training folder
    filename = f"training_data/game_{game_number}_{int(time.time())}.pgn"
    with open(filename, "w", encoding="utf-8") as f:
        exporter = chess.pgn.FileExporter(f)
        pgn_game.accept(exporter)

if __name__ == "__main__":
    print("Initializing Arena...")
    games_to_play = 50 # Let it play 50 games in a row
    
    for i in range(1, games_to_play + 1):
        play_self_match(i)
        
    print(f"\nSuccessfully generated {games_to_play} PGN files for training!")