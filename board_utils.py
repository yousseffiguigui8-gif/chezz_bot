import chess
import numpy as np

def board_to_tensor(board):
    """Converts a chess.Board to an 8x8x12 NumPy array."""
    tensor = np.zeros((8, 8, 12), dtype=np.float32)
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            # Calculate row and column
            row, col = divmod(square, 8)
            # Map piece to one of the 12 channels
            channel = piece.piece_type - 1
            if piece.color == chess.BLACK:
                channel += 6
            
            tensor[row, col, channel] = 1.0
            
    return tensor