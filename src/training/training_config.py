import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models


class TrainingConfig:
    def __init__(self, name: str):
        # Config name to help with identification
        self.model_name: str = name

        # Batch size for training
        self.batch_size: int = 32

        # Number of epochs to train for
        self.epochs: int = 10

        # Optimizer
        self.optimizer: tf.keras.optimizer = tf.keras.optimizers.Adam()

        # Loss function
        self.loss: tf.keras.loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),

        # Input shape
        self.input_h = 128  # Always 128
        self.input_w = 512  # Corresponds to the track's length; 512 is around 6 seconds

        # Mode layers definition
        self.model = models.Sequential([
            layers.Input(shape=(self.input_h, self.input_w)),
            layers.Resizing(32, 32),
            layers.Conv2D(32, 3, activation='relu'),
            layers.Conv2D(64, 3, activation='relu'),
            layers.MaxPooling2D(),
            layers.Dropout(0.25),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(8),
        ])
