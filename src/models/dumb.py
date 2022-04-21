from tensorflow.keras import layers
from tensorflow.keras import models

def dumb_model(input_shape, num_classes):
    model = models.Sequential()
    model.add(layers.Input(input_shape))

    model.add(layers.Conv2D(8, kernel_size=(3, 3), strides=(1, 1)))
    model.add(layers.BatchNormalization(axis=3))
    model.add(layers.Activation("relu"))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(16, kernel_size=(3, 3), strides=(1, 1)))
    model.add(layers.BatchNormalization(axis=3))
    model.add(layers.Activation("relu"))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(32, kernel_size=(3, 3), strides=(1, 1)))
    model.add(layers.BatchNormalization(axis=3))
    model.add(layers.Activation("relu"))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(64, kernel_size=(3, 3), strides=(1, 1)))
    model.add(layers.BatchNormalization(axis=-1))
    model.add(layers.Activation("relu"))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(128, kernel_size=(3, 3), strides=(1, 1)))
    model.add(layers.BatchNormalization(axis=-1))
    model.add(layers.Activation("relu"))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Flatten())

    model.add(layers.Dropout(rate=0.3))

    model.add(layers.Dense(num_classes, activation="softmax"))

    return model
