import chess
import chess.polyglot
import numpy as np
import tensorflow as tf
from board_utils import board_to_tensor
import random
import time

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


def ordered_moves(board):
    """Returns legal moves sorted by tactical priority."""
    moves = list(board.legal_moves)
    moves.sort(key=lambda move: score_move(board, move), reverse=True)
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
    try:
        with chess.polyglot.open_reader("book.bin") as reader:
            entry = reader.weighted_choice(board)
            return entry.move
    except Exception:
        return None

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

def minimax(board, depth, alpha, beta, maximizing_player, transposition_table, deadline=None):
    """The core alpha-beta pruning search tree."""
    if deadline is not None and time.perf_counter() >= deadline:
        raise SearchTimeout()

    if depth == 0 or board.is_game_over():
        # Drop into quiescence search instead of stopping blindly
        return quiescence_search(board, alpha, beta, deadline)

    key = transposition_key(board, maximizing_player)
    cached = transposition_table.get(key)
    alpha_original = alpha
    beta_original = beta

    if cached and cached["depth"] >= depth:
        cached_value = cached["value"]
        cached_flag = cached["flag"]

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
        for move in ordered_moves(board):
            board.push(move)
            eval = minimax(board, depth - 1, alpha, beta, False, transposition_table, deadline)
            board.pop()
            max_eval = max(max_eval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        value = max_eval
    else:
        min_eval = float('inf')
        for move in ordered_moves(board):
            board.push(move)
            eval = minimax(board, depth - 1, alpha, beta, True, transposition_table, deadline)
            board.pop()
            min_eval = min(min_eval, eval)
            beta = min(beta, eval)
            if beta <= alpha:
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
    }

    return value

def get_best_move(board, depth=2, time_limit=None):
    """Entry point for the bot to pick its move.

    Uses iterative deepening when time_limit is provided (seconds).
    """
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return None

    try:
        with chess.polyglot.open_reader("book.bin") as reader:
            move = reader.weighted_choice(board).move
            print("--- Book move played ---")
            return move
    except Exception:
        pass

    print("No book move. Thinking...")

    ordered_legal_moves = ordered_moves(board)
    fallback_move = ordered_legal_moves[0] if ordered_legal_moves else random.choice(legal_moves)

    transposition_table = {}
    best_move = fallback_move
    deadline = None if time_limit is None else time.perf_counter() + max(0.01, float(time_limit))
    max_depth = max(1, int(depth))

    for current_depth in range(1, max_depth + 1):
        current_best_move = None
        current_best_value = -float('inf') if board.turn == chess.WHITE else float('inf')

        try:
            for move in ordered_legal_moves:
                board.push(move)
                board_value = minimax(
                    board,
                    current_depth - 1,
                    -float('inf'),
                    float('inf'),
                    not board.turn,
                    transposition_table,
                    deadline,
                )
                board.pop()

                if board.turn == chess.WHITE:
                    if board_value > current_best_value:
                        current_best_value = board_value
                        current_best_move = move
                else:
                    if board_value < current_best_value:
                        current_best_value = board_value
                        current_best_move = move

            if current_best_move is not None:
                best_move = current_best_move

            if deadline is not None and time.perf_counter() >= deadline:
                break
        except SearchTimeout:
            if current_best_move is not None:
                best_move = current_best_move
            break

    return best_move