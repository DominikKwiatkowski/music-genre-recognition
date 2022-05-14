import tensorflow as tf
from tensorflow.keras import mixed_precision

from models.multi_scale_level_cnn import multi_scale_level_cnn
from models.dumb import dumb_model

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# Enable eager execution
# tf.compat.v1.enable_eager_execution()

# Equivalent to the two lines above
mixed_precision.set_global_policy('mixed_float16')


class TrainingConfig:
    def __init__(self):
        # Batch size for training
        self.batch_size: int = 8

        self.num_classes: int = 8

        # Number of epochs to train for
        self.starting_epoch: int = 0
        self.epochs: int = 100

        # Learning rate
        self.learning_rate: float = 0.01
        self.learning_rate_patience: int = 3
        self.learning_rate_decrease_multiplier: float = 0.5
        self.learning_rate_min: float = 0.0

        # Early stopping conditions
        self.early_stopping_patience: int = 10
        self.early_stopping_min_delta: float = 0.01

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
        # self.model = multi_scale_level_cnn(
        #     (self.input_h, self.input_w, 1),
        #     num_dense_blocks=3,
        #     num_conv_filters=32,
        #     num_classes=self.num_classes,
        # )

        self.model = dumb_model(
            (self.input_h, self.input_w, 1),
            num_classes=self.num_classes,
        )

        self.model.summary()
