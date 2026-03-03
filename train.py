import tensorflow as tf
import numpy as np
import chess
import chess.pgn
import chess.engine
import os
import json
from datetime import datetime
from model import create_chess_model
from board_utils import board_to_tensor


PGN_PATH = "lichess_elite_2020-08.pgn"
CACHE_DIR = "dataset_cache"
CACHE_X_FILE = os.path.join(CACHE_DIR, "X_data.npy")
CACHE_Y_FILE = os.path.join(CACHE_DIR, "y_data.npy")
CACHE_META_FILE = os.path.join(CACHE_DIR, "metadata.json")
MODEL_OUTPUT_PATH = os.getenv("MODEL_OUTPUT_PATH", "best_chess_model.keras")
STOCKFISH_PATH = os.getenv("STOCKFISH_PATH")
STOCKFISH_DEPTH = int(os.getenv("STOCKFISH_DEPTH", "10"))
STOCKFISH_LABEL_FRACTION = float(os.getenv("STOCKFISH_LABEL_FRACTION", "0.25"))


def configure_training_device():
    """Configures TensorFlow to use GPU when available."""
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        print("No GPU detected by TensorFlow. Training will run on CPU.")
        return False

    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    print(f"TensorFlow detected {len(gpus)} GPU(s): {gpus}")
    return True


def game_result_to_value(result):
    if result == "1-0":
        return 1.0
    if result == "0-1":
        return -1.0
    return 0.0


def board_material_value(board):
    piece_values = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
    }
    score = 0
    for piece_type, value in piece_values.items():
        score += len(board.pieces(piece_type, chess.WHITE)) * value
        score -= len(board.pieces(piece_type, chess.BLACK)) * value
    normalized = max(-1.0, min(1.0, score / 39.0))
    return float(normalized)


def blended_target(board, game_result_value):
    """Blend final game result with static material value for a stronger label."""
    result_term = game_result_value if board.turn == chess.WHITE else -game_result_value
    material_term = board_material_value(board) if board.turn == chess.WHITE else -board_material_value(board)
    return (0.8 * result_term) + (0.2 * material_term)


def cp_to_value(cp_score):
    return float(np.tanh(cp_score / 400.0))


def stockfish_target(board, sf_engine, depth):
    if sf_engine is None:
        return None

    try:
        info = sf_engine.analyse(board, chess.engine.Limit(depth=depth))
        score = info.get("score")
        if score is None:
            return None

        pov = score.pov(chess.WHITE)
        if pov.is_mate():
            mate_score = pov.mate()
            if mate_score is None:
                return None
            return 1.0 if mate_score > 0 else -1.0

        cp = pov.score(mate_score=10000)
        if cp is None:
            return None

        white_value = cp_to_value(cp)
        return white_value if board.turn == chess.WHITE else -white_value
    except Exception:
        return None


def warmup_cosine_schedule(total_steps, warmup_steps, base_lr=1e-3, min_lr=1e-5):
    def scheduler(step):
        if step < warmup_steps:
            return min_lr + (base_lr - min_lr) * (step / max(1, warmup_steps))

        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        cosine_decay = 0.5 * (1 + np.cos(np.pi * progress))
        return min_lr + (base_lr - min_lr) * cosine_decay

    return scheduler


def load_training_data_from_pgn(pgn_path, max_positions=120000, min_ply=8, sample_stride=2):
    """Loads board tensors and value targets from a PGN file."""
    if not os.path.exists(pgn_path):
        raise FileNotFoundError(f"PGN file not found: {pgn_path}")

    features = []
    labels = []
    game_count = 0
    sf_engine = None

    if STOCKFISH_PATH and os.path.exists(STOCKFISH_PATH):
        try:
            sf_engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
            print(f"Stockfish labeling enabled: {STOCKFISH_PATH} (depth={STOCKFISH_DEPTH})")
        except Exception as sf_error:
            print(f"Warning: could not start Stockfish ({sf_error}); falling back to blended labels.")
            sf_engine = None

    try:
        with open(pgn_path, "r", encoding="utf-8", errors="ignore") as pgn_file:
            while len(features) < max_positions:
                game = chess.pgn.read_game(pgn_file)
                if game is None:
                    break

                result_value = game_result_to_value(game.headers.get("Result", "*"))
                if result_value == 0.0 and game.headers.get("Result", "*") not in ("1/2-1/2",):
                    continue

                board = game.board()
                ply = 0
                for move in game.mainline_moves():
                    if ply >= min_ply and (ply % sample_stride == 0):
                        fallback_value = blended_target(board, result_value)
                        sf_value = None
                        if sf_engine is not None and np.random.random() < STOCKFISH_LABEL_FRACTION:
                            sf_value = stockfish_target(board, sf_engine, STOCKFISH_DEPTH)

                        label_value = (0.7 * sf_value + 0.3 * fallback_value) if sf_value is not None else fallback_value
                        features.append(board_to_tensor(board))
                        labels.append(label_value)

                    board.push(move)
                    ply += 1

                    if len(features) >= max_positions:
                        break

                game_count += 1
    finally:
        if sf_engine is not None:
            sf_engine.quit()

    if not features:
        raise ValueError("No valid positions were loaded from the PGN file.")

    print(f"Loaded {len(features)} positions from {game_count} games in {pgn_path}")
    return np.array(features, dtype=np.float32), np.array(labels, dtype=np.float32)


def load_or_build_dataset(pgn_path, max_positions):
    os.makedirs(CACHE_DIR, exist_ok=True)
    source_info = {
        "pgn_path": pgn_path,
        "pgn_size": os.path.getsize(pgn_path) if os.path.exists(pgn_path) else None,
        "pgn_mtime": os.path.getmtime(pgn_path) if os.path.exists(pgn_path) else None,
        "max_positions": max_positions,
        "stockfish_path": STOCKFISH_PATH,
        "stockfish_depth": STOCKFISH_DEPTH,
        "stockfish_fraction": STOCKFISH_LABEL_FRACTION,
    }

    if os.path.exists(CACHE_X_FILE) and os.path.exists(CACHE_Y_FILE) and os.path.exists(CACHE_META_FILE):
        with open(CACHE_META_FILE, "r", encoding="utf-8") as meta_file:
            cached_meta = json.load(meta_file)

        if cached_meta == source_info:
            print("Loading dataset from cache...")
            X_data = np.load(CACHE_X_FILE, mmap_mode="r")
            y_data = np.load(CACHE_Y_FILE, mmap_mode="r")
            return X_data, y_data

    X_data, y_data = load_training_data_from_pgn(pgn_path, max_positions=max_positions)
    print("Saving dataset cache...")
    np.save(CACHE_X_FILE, X_data)
    np.save(CACHE_Y_FILE, y_data)
    with open(CACHE_META_FILE, "w", encoding="utf-8") as meta_file:
        json.dump(source_info, meta_file, indent=2)
    return X_data, y_data

def train_pipeline():
    has_gpu = configure_training_device()
    print("Initializing Neural Network...")
    model = create_chess_model()
    
    print("Loading training data...")
    pgn_path = PGN_PATH
    sample_count = 120000 if has_gpu else 30000
    batch_size = 1024 if has_gpu else 128
    try:
        X_data, y_data = load_or_build_dataset(pgn_path, max_positions=sample_count)
    except Exception as e:
        print(f"Could not load PGN data ({e}). Falling back to random data.")
        X_data = np.random.rand(sample_count, 8, 8, 12).astype(np.float32)
        y_data = np.random.uniform(-1, 1, sample_count).astype(np.float32)

    effective_count = len(X_data)
    split_idx = int(effective_count * 0.9)
    permutation = np.random.permutation(effective_count)
    train_idx = permutation[:split_idx]
    val_idx = permutation[split_idx:]

    X_train, y_train = X_data[train_idx], y_data[train_idx]
    X_val, y_val = X_data[val_idx], y_data[val_idx]

    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(min(50000, len(X_train))).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    steps_per_epoch = max(1, len(X_train) // batch_size)
    total_steps = steps_per_epoch * 15
    warmup_steps = max(100, int(total_steps * 0.05))

    # Smart callbacks for better training
    callbacks = [
        tf.keras.callbacks.LearningRateScheduler(warmup_cosine_schedule(total_steps, warmup_steps), verbose=0),
        tf.keras.callbacks.ModelCheckpoint(MODEL_OUTPUT_PATH, monitor='val_loss', mode='min', save_best_only=True),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=6, restore_best_weights=True),
        tf.keras.callbacks.TensorBoard(log_dir="./logs"),
        tf.keras.callbacks.CSVLogger("training_log.csv", append=True),
    ]

    print(f"Dataset size: {len(X_data)} positions")
    print(f"Train size: {len(train_idx)} | Validation size: {len(val_idx)}")
    print(f"Batch size: {batch_size}")
    
    print("Starting Training Loop...")
    # A batch size of 128 or 256 is highly efficient here
    history = model.fit(train_dataset, epochs=15, validation_data=val_dataset, callbacks=callbacks)

    run_metadata = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "dataset_size": int(len(X_data)),
        "train_size": int(len(train_idx)),
        "val_size": int(len(val_idx)),
        "batch_size": int(batch_size),
        "pgn_path": pgn_path,
        "model_output_path": MODEL_OUTPUT_PATH,
        "stockfish_path": STOCKFISH_PATH,
        "stockfish_depth": STOCKFISH_DEPTH,
        "stockfish_label_fraction": STOCKFISH_LABEL_FRACTION,
        "best_val_loss": float(min(history.history.get("val_loss", [float("inf")]))),
    }
    with open("training_run_metadata.json", "w", encoding="utf-8") as metadata_file:
        json.dump(run_metadata, metadata_file, indent=2)

    print("Pipeline Complete! Best model saved.")

if __name__ == "__main__":
    train_pipeline()