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
    parser.add_argument("--swap-colors", action="store_true", help="Play half games with swapped white/black models")
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


def evaluate_match(model_a, model_b, games, depth, time_limit, disable_book=False, swap_colors=False):
    results = {"1-0": 0, "0-1": 0, "1/2-1/2": 0}
    games_pgn = []

    for game_index in range(1, games + 1):
        if swap_colors and (game_index % 2 == 0):
            white_model = model_b
            black_model = model_a
            swap = True
        else:
            white_model = model_a
            black_model = model_b
            swap = False

        print(f"Playing game {game_index}/{games}...")
        result, game = play_game(white_model, black_model, depth, time_limit, disable_book)
        results[result] = results.get(result, 0) + 1

        if swap:
            if result == "1-0":
                score_for_a = 0.0
            elif result == "0-1":
                score_for_a = 1.0
            else:
                score_for_a = 0.5
            game.headers["White"] = "ModelB"
            game.headers["Black"] = "ModelA"
        else:
            if result == "1-0":
                score_for_a = 1.0
            elif result == "0-1":
                score_for_a = 0.0
            else:
                score_for_a = 0.5
            game.headers["White"] = "ModelA"
            game.headers["Black"] = "ModelB"

        game.headers["ScoreForA"] = str(score_for_a)
        games_pgn.append(game)

    score_for_a = sum(float(g.headers.get("ScoreForA", "0.5")) for g in games_pgn)
    return score_for_a, results, games_pgn


def main():
    args = parse_args()

    model_white = load_model(args.model_white)
    model_black = load_model(args.model_black)

    score_white, results, games = evaluate_match(
        model_white,
        model_black,
        args.games,
        args.depth,
        args.time_limit,
        disable_book=args.disable_book,
        swap_colors=args.swap_colors,
    )

    with open(args.output_pgn, "w", encoding="utf-8") as pgn_file:
        for game in games:
            print(game, file=pgn_file, end="\n\n")

    total = max(1, args.games)
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
