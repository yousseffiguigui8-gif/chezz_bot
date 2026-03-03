import tensorflow as tf
import numpy as np
from model import create_chess_model


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

def train_pipeline():
    has_gpu = configure_training_device()
    print("Initializing Neural Network...")
    model = create_chess_model()
    
    # Smart callbacks for better training
    callbacks = [
        # Automatically lower learning rate if the model stops improving
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1),
        # Only save the weights if they are better than the previous epoch
        tf.keras.callbacks.ModelCheckpoint("best_chess_model.keras", monitor='val_loss', mode='min', save_best_only=True),
        # Stop early if validation quality stalls and keep the best weights
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=6, restore_best_weights=True),
        # Enable TensorBoard tracking
        tf.keras.callbacks.TensorBoard(log_dir="./logs")
    ]
    
    # --- DATA LOADING PLACEHOLDER ---
    # For now, we generate random dummy data to test the pipeline.
    # You will replace this later with your PGN parsing script.
    print("Loading training data...")
    sample_count = 50000 if has_gpu else 5000
    batch_size = 1024 if has_gpu else 128
    X_data = np.random.rand(sample_count, 8, 8, 12).astype(np.float32)
    y_data = np.random.uniform(-1, 1, sample_count).astype(np.float32)

    split_idx = int(sample_count * 0.9)
    permutation = np.random.permutation(sample_count)
    train_idx = permutation[:split_idx]
    val_idx = permutation[split_idx:]

    X_train, y_train = X_data[train_idx], y_data[train_idx]
    X_val, y_val = X_data[val_idx], y_data[val_idx]

    print(f"Dataset size: {sample_count} positions")
    print(f"Train size: {len(train_idx)} | Validation size: {len(val_idx)}")
    print(f"Batch size: {batch_size}")
    
    print("Starting Training Loop...")
    # A batch size of 128 or 256 is highly efficient here
    model.fit(
        X_train, y_train, 
        epochs=15, 
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        callbacks=callbacks
    )
    print("Pipeline Complete! Best model saved.")

if __name__ == "__main__":
    train_pipeline()