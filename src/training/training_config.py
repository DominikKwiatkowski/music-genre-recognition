import tensorflow as tf

from models.multi_scale_level_cnn import multi_scale_level_cnn
from models.dumb import dumb_model


class TrainingConfig:
    def __init__(self):
        # Batch size for training
        self.batch_size: int = 8

        self.starting_epoch: int = 0
        # Number of epochs to train for
        self.epochs: int = 100

        # Learning rate
        self.learning_rate: float = 0.01
        self.learning_rate_patience: int = 3

        # Optimizer
        self.optimizer: tf.keras.optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.learning_rate
        )

        # Loss function
        self.loss: tf.keras.loss = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=False
        )

        # Input shape
        self.input_h = 128  # Always 128
        self.input_w = 512  # Corresponds to the track's length; 512 is around 6 seconds
        self.patch_size = 3276800 / self.input_w  # size which can be allocated on GPU
        # Mode layers definition
        self.model = dumb_model(
            (self.input_h, self.input_w, 1),
            num_classes=8,
        )

        self.model.summary()
