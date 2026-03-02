import chess
import numpy as np
import tensorflow as tf
from board_utils import board_to_tensor
import random

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

def quiescence_search(board, alpha, beta):
    """Continues searching until all captures are resolved (No Horizon Effect)."""
    stand_pat = get_neural_evaluation(board)
    
    if stand_pat >= beta:
        return beta
    if alpha < stand_pat:
        alpha = stand_pat
        
    for move in board.legal_moves:
        if board.is_capture(move):
            board.push(move)
            score = -quiescence_search(board, -beta, -alpha)
            board.pop()
            
            if score >= beta:
                return beta
            if score > alpha:
                alpha = score
    return alpha

def minimax(board, depth, alpha, beta, maximizing_player):
    """The core alpha-beta pruning search tree."""
    if depth == 0 or board.is_game_over():
        # Drop into quiescence search instead of stopping blindly
        return quiescence_search(board, alpha, beta)

    if maximizing_player:
        max_eval = -float('inf')
        for move in board.legal_moves:
            board.push(move)
            eval = minimax(board, depth - 1, alpha, beta, False)
            board.pop()
            max_eval = max(max_eval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        return max_eval
    else:
        min_eval = float('inf')
        for move in board.legal_moves:
            board.push(move)
            eval = minimax(board, depth - 1, alpha, beta, True)
            board.pop()
            min_eval = min(min_eval, eval)
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return min_eval

def get_best_move(board, depth=2):
    """Entry point for the bot to pick its move."""
    best_move = None
    best_value = -float('inf') if board.turn == chess.WHITE else float('inf')
    
    for move in board.legal_moves:
        board.push(move)
        board_value = minimax(board, depth - 1, -float('inf'), float('inf'), not board.turn)
        board.pop()

        if board.turn == chess.WHITE:
            if board_value > best_value:
                best_value = board_value
                best_move = move
        else:
            if board_value < best_value:
                best_value = board_value
                best_move = move

    # Fallback if everything looks equally bad
    if best_move is None:
        best_move = random.choice(list(board.legal_moves))
        
    return best_move