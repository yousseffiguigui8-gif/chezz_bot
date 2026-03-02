import tensorflow as tf

def build_residual_block(inputs, filters):
    """Creates a standard residual block with skip connections."""
    x = tf.keras.layers.Conv2D(filters, 3, padding="same", use_bias=False)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    
    x = tf.keras.layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    
    # Skip connection (The 'Residual' part)
    x = tf.keras.layers.Add()([inputs, x])
    return tf.keras.layers.Activation("relu")(x)

def create_chess_model():
    """Builds and compiles the CNN."""
    # Hardware Optimization: Force 16-bit precision for faster training
    tf.keras.mixed_precision.set_global_policy('mixed_float16')

    inputs = tf.keras.Input(shape=(8, 8, 12))
    
    # Initial Convolution
    x = tf.keras.layers.Conv2D(64, 3, padding="same", use_bias=False)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    
    # Residual Tower (5 blocks is a safe, powerful starting point)
    for _ in range(5):
        x = build_residual_block(x, 64)
        
    # Value Head (Evaluates the board state)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    
    # Output must be float32 for numerical stability in the loss function
    outputs = tf.keras.layers.Dense(1, activation="tanh", dtype='float32')(x) 
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
        loss="mse", 
        metrics=["mae"]
    )
    return model