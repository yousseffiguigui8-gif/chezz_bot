import berserk
import chess
import engine
import threading
import os
import time
from datetime import timedelta


def phase_depth(board):
    piece_count = len(board.piece_map())
    if piece_count <= 12:
        return 6
    if piece_count <= 20:
        return 5
    return 4


def to_seconds(value, default_seconds):
    if value is None:
        return default_seconds
    if isinstance(value, timedelta):
        return max(0.0, value.total_seconds())
    if isinstance(value, (int, float)):
        if value > 1000:
            return max(0.0, float(value) / 1000.0)
        return max(0.0, float(value))
    return default_seconds


def compute_time_limit(state, my_color, board):
    if not state or my_color is None:
        return 1.5

    if my_color == chess.WHITE:
        remaining_ms = state.get("wtime", 60000)
        increment_ms = state.get("winc", 0)
    else:
        remaining_ms = state.get("btime", 60000)
        increment_ms = state.get("binc", 0)

    remaining = to_seconds(remaining_ms, 60.0)
    increment = to_seconds(increment_ms, 0.0)
    pieces = len(board.piece_map())

    if remaining < 10:
        return min(0.25, max(0.08, increment * 0.3 + 0.05))
    if remaining < 30:
        return min(0.8, max(0.2, increment * 0.5 + 0.2))

    if pieces > 24:
        divisor = 36
    elif pieces > 14:
        divisor = 28
    else:
        divisor = 22

    allocation = (remaining / divisor) + (increment * 0.7)
    return min(5.0, max(0.2, allocation))



def make_move(game_id, board, client, state=None, my_color=None):
    """Asks the neural network engine for the best move and sends it to Lichess."""
    print("Bot is thinking...")
    depth = phase_depth(board)
    time_limit = compute_time_limit(state, my_color, board)
    best_move = engine.get_best_move(board, depth=depth, time_limit=time_limit)
    
    if best_move:
        print(f"Playing move: {best_move.uci()}")
        try:
            client.bots.make_move(game_id, best_move.uci())
        except Exception as e:
            print(f"Error making move: {e}")
    else:
        print("Bot resigned or no legal moves.")

def play_game(game_id, client):
    """Handles the live game loop for a specific Lichess match."""
    board = chess.Board()
    my_color = None
    
    print(f"\n--- Game {game_id} started! ---")
    
    try:
        # Get bot's Lichess username to figure out what color we are playing
        bot_account = client.account.get()
        bot_username = bot_account['username']
    except Exception as e:
        print(f"Could not fetch account info: {e}")
        return

    try:
        for event in client.bots.stream_game_state(game_id):
            if event['type'] == 'gameFull':
                white_player = event['white'].get('id', '')
                if white_player == bot_username.lower():
                    my_color = chess.WHITE
                    print("Playing as WHITE")
                else:
                    my_color = chess.BLACK
                    print("Playing as BLACK")

                state = event['state']
                moves = state['moves'].split() if state['moves'] else []
                for m in moves:
                    board.push_uci(m)

                if board.turn == my_color and not board.is_game_over():
                    make_move(game_id, board, client, state=state, my_color=my_color)

            elif event['type'] == 'gameState':
                state = event
                moves = event['moves'].split() if event['moves'] else []
                board.clear_board()
                board.set_fen(chess.STARTING_FEN)
                for m in moves:
                    board.push_uci(m)

                if board.turn == my_color and not board.is_game_over():
                    make_move(game_id, board, client, state=state, my_color=my_color)
    except Exception as e:
        print(f"Game stream error for {game_id}: {e}")

def main():
    lichess_token = os.getenv("LICHESS_TOKEN")
    if not lichess_token:
        raise RuntimeError("Set the LICHESS_TOKEN environment variable before starting the bot.")

    session = berserk.TokenSession(lichess_token)
    client = berserk.Client(session)

    try:
        # Automatically upgrade the account to BOT status (Only does something the first time)
        client.account.upgrade_to_bot()
        print("Account upgraded to BOT status!")
    except Exception:
        # If it fails, it usually just means the account is already a BOT, which is fine.
        pass

    print("Listening for Lichess challenges...")

    retry_delay = 2
    while True:
        try:
            for event in client.bots.stream_incoming_events():
                if event['type'] == 'challenge':
                    challenge_id = event['challenge']['id']
                    challenger = event['challenge']['challenger']['name']
                    print(f"Accepting challenge from {challenger} (ID: {challenge_id})...")
                    client.bots.accept_challenge(challenge_id)

                elif event['type'] == 'gameStart':
                    game_id = event['game']['id']
                    thread = threading.Thread(target=play_game, args=(game_id, client), daemon=True)
                    thread.start()

            retry_delay = 2
        except Exception as e:
            print(f"Connection error: {e}. Reconnecting in {retry_delay}s...")
            time.sleep(retry_delay)
            retry_delay = min(retry_delay * 2, 60)

if __name__ == "__main__":
    main()