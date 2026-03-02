import tensorflow as tf
import numpy as np
from model import create_chess_model

def train_pipeline():
    print("Initializing Neural Network...")
    model = create_chess_model()
    
    # Smart callbacks for better training
    callbacks = [
        # Automatically lower learning rate if the model stops improving
        tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=3, verbose=1),
        # Only save the weights if they are better than the previous epoch
        tf.keras.callbacks.ModelCheckpoint("best_chess_model.keras", save_best_only=True),
        # Enable TensorBoard tracking
        tf.keras.callbacks.TensorBoard(log_dir="./logs")
    ]
    
    # --- DATA LOADING PLACEHOLDER ---
    # For now, we generate random dummy data to test the pipeline.
    # You will replace this later with your PGN parsing script.
    print("Loading training data...")
    X_train = np.random.rand(1000, 8, 8, 12).astype(np.float32) 
    y_train = np.random.uniform(-1, 1, 1000).astype(np.float32)
    
    print("Starting Training Loop...")
    # A batch size of 128 or 256 is highly efficient here
    model.fit(
        X_train, y_train, 
        epochs=15, 
        batch_size=128, 
        callbacks=callbacks
    )
    print("Pipeline Complete! Best model saved.")

if __name__ == "__main__":
    train_pipeline()