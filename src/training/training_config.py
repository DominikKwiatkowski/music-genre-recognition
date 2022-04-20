import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models


class TrainingConfig:
    def __init__(self, name: str):
        # Config name to help with identification
        self.model_name: str = name

        # Batch size for training
        self.batch_size: int = 64

        # Number of epochs to train for
        self.epochs: int = 5

        # Optimizer
        self.optimizer: tf.keras.optimizer = tf.keras.optimizers.Adam()

        # Loss function
        self.loss: tf.keras.loss = (
            tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        )

        # Input shape
        self.input_h = 128  # Always 128
        self.input_w = 512  # Corresponds to the track's length; 512 is around 6 seconds
        # Mode layers definition
        self.model = models.Sequential()
        self.model.add(layers.Input((self.input_h, self.input_w, 1)))

        self.model.add(layers.Conv2D(8, kernel_size=(3, 3), strides=(1, 1)))
        self.model.add(layers.BatchNormalization(axis=3))
        self.model.add(layers.Activation("relu"))
        self.model.add(layers.MaxPooling2D((2, 2)))

        self.model.add(layers.Conv2D(16, kernel_size=(3, 3), strides=(1, 1)))
        self.model.add(layers.BatchNormalization(axis=3))
        self.model.add(layers.Activation("relu"))
        self.model.add(layers.MaxPooling2D((2, 2)))

        self.model.add(layers.Conv2D(32, kernel_size=(3, 3), strides=(1, 1)))
        self.model.add(layers.BatchNormalization(axis=3))
        self.model.add(layers.Activation("relu"))
        self.model.add(layers.MaxPooling2D((2, 2)))

        self.model.add(layers.Conv2D(64, kernel_size=(3, 3), strides=(1, 1)))
        self.model.add(layers.BatchNormalization(axis=-1))
        self.model.add(layers.Activation("relu"))
        self.model.add(layers.MaxPooling2D((2, 2)))

        self.model.add(layers.Conv2D(128, kernel_size=(3, 3), strides=(1, 1)))
        self.model.add(layers.BatchNormalization(axis=-1))
        self.model.add(layers.Activation("relu"))
        self.model.add(layers.MaxPooling2D((2, 2)))

        self.model.add(layers.Flatten())

        self.model.add(layers.Dropout(rate=0.3))

        self.model.add(layers.Dense(8, activation="softmax"))

        self.model.summary()
