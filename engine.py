import chess
import chess.polyglot
import chess.syzygy
import numpy as np
import tensorflow as tf
from board_utils import board_to_tensor
import random
import time
import os

PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 20000,
}

TT_EXACT = "exact"
TT_LOWER = "lower"
TT_UPPER = "upper"


def _env_flag(name, default=False):
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


BOOK_PATH = os.getenv("CHESS_BOOK_PATH", "book.bin")
BOOK_MAX_PLY = int(os.getenv("CHESS_BOOK_MAX_PLY", "24"))
BOOK_WEIGHTED = _env_flag("CHESS_BOOK_WEIGHTED", default=True)
TABLEBASE_PATH = os.getenv("CHESS_TABLEBASE_PATH", "tablebases")
USE_TABLEBASE = _env_flag("CHESS_USE_TABLEBASE", default=True)
ASPIRATION_WINDOW = float(os.getenv("CHESS_ASPIRATION_WINDOW", "0.20"))

TABLEBASE = None
if USE_TABLEBASE and os.path.isdir(TABLEBASE_PATH):
    try:
        TABLEBASE = chess.syzygy.open_tablebase(TABLEBASE_PATH)
        print(f"Syzygy tablebase enabled: {TABLEBASE_PATH}")
    except Exception as tablebase_error:
        print(f"Warning: Could not open tablebases at {TABLEBASE_PATH}. Error: {tablebase_error}")


class SearchTimeout(Exception):
    """Raised when the search exceeds the allocated move time."""
    pass

# Load the brain (Ensure you run train.py at least once to create this file)
try:
    model = tf.keras.models.load_model("best_chess_model.keras")
    print("Brain successfully loaded into the Engine.")
except Exception as e:
    print(f"Warning: Could not load model. Error: {e}")
    model = None

def get_neural_evaluation(board):
    """Asks the CNN for the positional score."""
    if model is None:
        return 0.0
    
    tensor = board_to_tensor(board)
    tensor_batch = np.expand_dims(tensor, axis=0)
    # training=False is critical for fast GPU inference
    return float(model(tensor_batch, training=False)[0][0])


def score_move(board, move):
    """Prioritize tactical and forcing moves to improve alpha-beta pruning."""
    score = 0

    if board.is_capture(move):
        attacker_piece = board.piece_at(move.from_square)

        if board.is_en_passant(move):
            victim_value = PIECE_VALUES[chess.PAWN]
        else:
            victim_piece = board.piece_at(move.to_square)
            victim_value = PIECE_VALUES[victim_piece.piece_type] if victim_piece else 0

        attacker_value = PIECE_VALUES[attacker_piece.piece_type] if attacker_piece else 0
        score += 10000 + (victim_value * 10) - attacker_value

    if move.promotion:
        score += 8000 + PIECE_VALUES.get(move.promotion, 0)

    if board.gives_check(move):
        score += 2000

    return score


def ordered_moves(board, killer_moves=None, history_table=None, ply=0, pv_move=None):
    """Returns legal moves sorted by tactical priority."""
    moves = list(board.legal_moves)

    killers = killer_moves.get(ply, []) if killer_moves else []

    def move_priority(move):
        priority = score_move(board, move)

        if pv_move is not None and move == pv_move:
            priority += 200000
        if move in killers:
            priority += 15000
        if history_table:
            priority += history_table.get((move.from_square, move.to_square), 0)

        return priority

    moves.sort(key=move_priority, reverse=True)
    return moves


def transposition_key(board, maximizing_player):
    """Builds a hashable key for the current search node."""
    return (
        board.board_fen(),
        board.turn,
        board.castling_rights,
        board.ep_square,
        maximizing_player,
    )


def get_book_move(board):
    """Returns a Polyglot opening move when available."""
    if board.ply() > BOOK_MAX_PLY:
        return None

    try:
        with chess.polyglot.open_reader(BOOK_PATH) as reader:
            if BOOK_WEIGHTED:
                entry = reader.weighted_choice(board)
            else:
                entry = reader.find(board)
            return entry.move
    except Exception:
        return None


def probe_tablebase_value(board):
    """Returns a perfect endgame value from Syzygy if available."""
    if TABLEBASE is None:
        return None

    if len(board.piece_map()) > 7:
        return None

    try:
        wdl = TABLEBASE.probe_wdl(board)
    except Exception:
        return None

    if wdl > 0:
        return 1.0
    if wdl < 0:
        return -1.0
    return 0.0

def quiescence_search(board, alpha, beta, deadline=None):
    """Continues searching until all captures are resolved (No Horizon Effect)."""
    if deadline is not None and time.perf_counter() >= deadline:
        raise SearchTimeout()

    stand_pat = get_neural_evaluation(board)
    
    if stand_pat >= beta:
        return beta
    if alpha < stand_pat:
        alpha = stand_pat
        
    capture_moves = [move for move in ordered_moves(board) if board.is_capture(move)]
    for move in capture_moves:
        board.push(move)
        score = -quiescence_search(board, -beta, -alpha, deadline)
        board.pop()

        if score >= beta:
            return beta
        if score > alpha:
            alpha = score
    return alpha

def minimax(board, depth, alpha, beta, maximizing_player, transposition_table, killer_moves, history_table, ply=0, deadline=None):
    """The core alpha-beta pruning search tree."""
    if deadline is not None and time.perf_counter() >= deadline:
        raise SearchTimeout()

    tablebase_value = probe_tablebase_value(board)
    if tablebase_value is not None:
        return tablebase_value

    if depth == 0 or board.is_game_over():
        # Drop into quiescence search instead of stopping blindly
        return quiescence_search(board, alpha, beta, deadline)

    key = transposition_key(board, maximizing_player)
    cached = transposition_table.get(key)
    alpha_original = alpha
    beta_original = beta

    pv_move = None
    if cached and cached["depth"] >= depth:
        cached_value = cached["value"]
        cached_flag = cached["flag"]
        pv_move = cached.get("best_move")

        if cached_flag == TT_EXACT:
            return cached_value
        if cached_flag == TT_LOWER:
            alpha = max(alpha, cached_value)
        elif cached_flag == TT_UPPER:
            beta = min(beta, cached_value)

        if alpha >= beta:
            return cached_value

    if maximizing_player:
        max_eval = -float('inf')
        best_move_node = None
        ordered = ordered_moves(board, killer_moves, history_table, ply=ply, pv_move=pv_move)
        for index, move in enumerate(ordered):
            board.push(move)

            if index == 0:
                eval = minimax(board, depth - 1, alpha, beta, False, transposition_table, killer_moves, history_table, ply + 1, deadline)
            else:
                eval = minimax(board, depth - 1, alpha, alpha + 1e-6, False, transposition_table, killer_moves, history_table, ply + 1, deadline)
                if alpha < eval < beta:
                    eval = minimax(board, depth - 1, alpha, beta, False, transposition_table, killer_moves, history_table, ply + 1, deadline)

            board.pop()

            if eval > max_eval:
                max_eval = eval
                best_move_node = move

            alpha = max(alpha, eval)
            if beta <= alpha:
                if not board.is_capture(move):
                    killer_moves.setdefault(ply, [])
                    if move not in killer_moves[ply]:
                        killer_moves[ply] = (killer_moves[ply] + [move])[-2:]
                    key = (move.from_square, move.to_square)
                    history_table[key] = history_table.get(key, 0) + depth * depth
                break
        value = max_eval
    else:
        min_eval = float('inf')
        best_move_node = None
        ordered = ordered_moves(board, killer_moves, history_table, ply=ply, pv_move=pv_move)
        for index, move in enumerate(ordered):
            board.push(move)

            if index == 0:
                eval = minimax(board, depth - 1, alpha, beta, True, transposition_table, killer_moves, history_table, ply + 1, deadline)
            else:
                eval = minimax(board, depth - 1, beta - 1e-6, beta, True, transposition_table, killer_moves, history_table, ply + 1, deadline)
                if alpha < eval < beta:
                    eval = minimax(board, depth - 1, alpha, beta, True, transposition_table, killer_moves, history_table, ply + 1, deadline)

            board.pop()

            if eval < min_eval:
                min_eval = eval
                best_move_node = move

            beta = min(beta, eval)
            if beta <= alpha:
                if not board.is_capture(move):
                    killer_moves.setdefault(ply, [])
                    if move not in killer_moves[ply]:
                        killer_moves[ply] = (killer_moves[ply] + [move])[-2:]
                    key = (move.from_square, move.to_square)
                    history_table[key] = history_table.get(key, 0) + depth * depth
                break
        value = min_eval

    if value <= alpha_original:
        flag = TT_UPPER
    elif value >= beta_original:
        flag = TT_LOWER
    else:
        flag = TT_EXACT

    transposition_table[key] = {
        "depth": depth,
        "value": value,
        "flag": flag,
        "best_move": best_move_node,
    }

    return value

def get_best_move(board, depth=2, time_limit=None):
    """Entry point for the bot to pick its move.

    Uses iterative deepening when time_limit is provided (seconds).
    """
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return None

    book_move = get_book_move(board)
    if book_move:
        print("--- Book move played ---")
        return book_move

    print("No book move. Thinking...")

    ordered_legal_moves = ordered_moves(board)
    fallback_move = ordered_legal_moves[0] if ordered_legal_moves else random.choice(legal_moves)

    transposition_table = {}
    killer_moves = {}
    history_table = {}
    best_move = fallback_move
    deadline = None if time_limit is None else time.perf_counter() + max(0.01, float(time_limit))
    max_depth = max(1, int(depth))

    for current_depth in range(1, max_depth + 1):
        current_best_move = None
        current_best_value = -float('inf') if board.turn == chess.WHITE else float('inf')
        last_best_value = current_best_value

        try:
            alpha = -float('inf')
            beta = float('inf')
            if current_depth > 1 and np.isfinite(last_best_value):
                alpha = last_best_value - ASPIRATION_WINDOW
                beta = last_best_value + ASPIRATION_WINDOW

            def evaluate_root(search_alpha, search_beta):
                nonlocal current_best_move, current_best_value
                current_best_move = None
                current_best_value = -float('inf') if board.turn == chess.WHITE else float('inf')

                local_alpha = search_alpha
                local_beta = search_beta

                for move in ordered_legal_moves:
                    board.push(move)
                    board_value = minimax(
                        board,
                        current_depth - 1,
                        local_alpha,
                        local_beta,
                        not board.turn,
                        transposition_table,
                        killer_moves,
                        history_table,
                        1,
                        deadline,
                    )
                    board.pop()

                    if board.turn == chess.WHITE:
                        if board_value > current_best_value:
                            current_best_value = board_value
                            current_best_move = move
                        local_alpha = max(local_alpha, board_value)
                    else:
                        if board_value < current_best_value:
                            current_best_value = board_value
                            current_best_move = move
                        local_beta = min(local_beta, board_value)

            evaluate_root(alpha, beta)

            if (current_best_value <= alpha) or (current_best_value >= beta):
                evaluate_root(-float('inf'), float('inf'))

            if current_best_move is not None:
                best_move = current_best_move
                last_best_value = current_best_value

            if deadline is not None and time.perf_counter() >= deadline:
                break
        except SearchTimeout:
            if current_best_move is not None:
                best_move = current_best_move
            break

    return best_move