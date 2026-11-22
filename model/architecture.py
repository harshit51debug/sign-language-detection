import tensorflow as tf
from tensorflow.keras import layers, models

def build_sign_classifier(
    input_dim: int = 63,
    num_classes: int = 26,
    dropout_rate: float = 0.3
) -> models.Model:
    """
    Builds a regularized Deep Neural Network (MLP) for classifying
    normalized hand landmark coordinates into 26 sign language alphabet classes.
    
    Architecture:
      Input (63) -> Dense(256) -> BatchNorm -> ReLU -> Dropout(0.3)
                 -> Dense(128) -> BatchNorm -> ReLU -> Dropout(0.3)
                 -> Dense(64)  -> BatchNorm -> ReLU -> Dropout(0.2)
                 -> Dense(num_classes) -> Softmax
    """
    inputs = layers.Input(shape=(input_dim,), name="landmark_input")
    
    # Layer 1
    x = layers.Dense(256, kernel_regularizer=tf.keras.regularizers.l2(1e-4))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(dropout_rate)(x)

    # Layer 2
    x = layers.Dense(128, kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(dropout_rate)(x)

    # Layer 3
    x = layers.Dense(64, kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.2)(x)

    # Output Layer
    outputs = layers.Dense(num_classes, activation="softmax", name="class_probabilities")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="SignLanguageClassifier")
    return model

if __name__ == "__main__":
    test_model = build_sign_classifier(input_dim=63, num_classes=26)
    test_model.summary()
