import argparse
import chess
import engine


def parse_args():
    parser = argparse.ArgumentParser(description="Play a local game against the Chezz bot.")
    parser.add_argument("--color", choices=["white", "black"], default="white", help="Your side")
    parser.add_argument("--depth", type=int, default=5, help="Bot search depth")
    parser.add_argument("--time-limit", type=float, default=1.5, help="Bot time per move in seconds")
    return parser.parse_args()


def get_human_move(board):
    while True:
        user_input = input("Your move (UCI like e2e4, or 'quit'): ").strip().lower()
        if user_input in {"quit", "exit"}:
            return None

        try:
            move = chess.Move.from_uci(user_input)
        except ValueError:
            print("Invalid format. Use UCI notation like e2e4 or g1f3.")
            continue

        if move not in board.legal_moves:
            print("Illegal move in this position. Try again.")
            continue

        return move


def main():
    args = parse_args()
    human_color = chess.WHITE if args.color == "white" else chess.BLACK
    board = chess.Board()

    print("Starting local game. Type moves in UCI format (example: e2e4).")
    print(board)

    while not board.is_game_over(claim_draw=True):
        if board.turn == human_color:
            move = get_human_move(board)
            if move is None:
                print("Game ended by user.")
                return
            board.push(move)
            print(f"You played: {move.uci()}")
        else:
            print("Bot is thinking...")
            bot_move = engine.get_best_move(board, depth=args.depth, time_limit=args.time_limit)
            if bot_move is None:
                print("Bot has no legal moves.")
                break
            board.push(bot_move)
            print(f"Bot played: {bot_move.uci()}")

        print(board)
        print()

    print(f"Game over: {board.result(claim_draw=True)}")


if __name__ == "__main__":
    main()
