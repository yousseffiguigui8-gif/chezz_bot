import argparse
import random
import chess
import chess.pgn
import tensorflow as tf
import engine


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate two models by playing head-to-head games.")
    parser.add_argument("--model-white", default="best_chess_model.keras", help="Model path for white side")
    parser.add_argument("--model-black", default="best_chess_model.keras", help="Model path for black side")
    parser.add_argument("--games", type=int, default=10, help="Number of games")
    parser.add_argument("--depth", type=int, default=3, help="Search depth")
    parser.add_argument("--time-limit", type=float, default=1.0, help="Seconds per move")
    parser.add_argument("--disable-book", action="store_true", help="Disable opening book usage during evaluation")
    parser.add_argument("--output-pgn", default="evaluation_games.pgn", help="Where to save played games")
    return parser.parse_args()


def load_model(path):
    print(f"Loading model: {path}")
    return tf.keras.models.load_model(path)


def select_move(board, model, depth, time_limit):
    original_model = engine.model
    try:
        engine.model = model
        return engine.get_best_move(board, depth=depth, time_limit=time_limit)
    finally:
        engine.model = original_model


def play_game(model_white, model_black, depth, time_limit, disable_book):
    board = chess.Board()
    game = chess.pgn.Game()
    game.headers["Event"] = "Chezz Model Evaluation"
    game.headers["White"] = "ModelWhite"
    game.headers["Black"] = "ModelBlack"

    node = game

    if disable_book:
        original_book_ply = engine.BOOK_MAX_PLY
        engine.BOOK_MAX_PLY = 0
    else:
        original_book_ply = None

    try:
        while not board.is_game_over():
            active_model = model_white if board.turn == chess.WHITE else model_black
            move = select_move(board, active_model, depth, time_limit)
            if move is None:
                break
            board.push(move)
            node = node.add_variation(move)
    finally:
        if original_book_ply is not None:
            engine.BOOK_MAX_PLY = original_book_ply

    game.headers["Result"] = board.result(claim_draw=True)
    return board.result(claim_draw=True), game


def main():
    args = parse_args()

    model_white = load_model(args.model_white)
    model_black = load_model(args.model_black)

    results = {"1-0": 0, "0-1": 0, "1/2-1/2": 0}
    games = []

    for game_index in range(1, args.games + 1):
        print(f"Playing game {game_index}/{args.games}...")
        result, game = play_game(model_white, model_black, args.depth, args.time_limit, args.disable_book)
        results[result] = results.get(result, 0) + 1
        games.append(game)

    with open(args.output_pgn, "w", encoding="utf-8") as pgn_file:
        for game in games:
            print(game, file=pgn_file, end="\n\n")

    total = max(1, args.games)
    score_white = results["1-0"] + 0.5 * results["1/2-1/2"]
    win_rate_white = score_white / total

    print("\n=== Evaluation Summary ===")
    print(f"White wins: {results['1-0']}")
    print(f"Black wins: {results['0-1']}")
    print(f"Draws:      {results['1/2-1/2']}")
    print(f"White score: {score_white:.1f}/{total} ({win_rate_white * 100:.1f}%)")
    print(f"Games saved to: {args.output_pgn}")


if __name__ == "__main__":
    random.seed(42)
    main()
