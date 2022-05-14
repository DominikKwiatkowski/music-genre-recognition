import os
from typing import Tuple, List

import pandas as pd
import tensorflow as tf
import numpy as np

from training.training_config import TrainingConfig
from sklearn import preprocessing

label_encoder = preprocessing.LabelEncoder()


def load_spectrogram_from_path(image_path, image_label, data_w):
    spectrogram = np.load(image_path)
    starting_index = np.random.randint(0, spectrogram.shape[1] - data_w)
    spectrogram = spectrogram[:, starting_index : starting_index + data_w]

    # return the image and the integer encoded label
    return spectrogram, image_label


def _fixup_shape(images, labels):
    images.set_shape([128, 512])
    labels.set_shape([])  # I have 8 classes
    return images, labels


def get_image_paths_and_labels(
    metadata: pd.DataFrame,
    path: str,
    training_config: TrainingConfig,
) -> Tuple[List[str], List[int]]:
    paths_list: List[str] = []
    labels_list: List[str] = []

    for row in metadata.iterrows():
        # TODO: Move validation of data to separate module
        try:
            # Load the data
            spectrogram_path = os.path.join(path, f"{row[1]['track_id']}.npy")
            spectrogram_data = np.load(spectrogram_path)

            # Validate the data before adding it to the training set
            if spectrogram_data.shape[1] > training_config.input_w:
                paths_list.append(spectrogram_path)
                labels_list.append(row[1]["genre_top"])
        except (FileNotFoundError, ValueError):
            pass

    return paths_list, label_encoder.fit_transform(labels_list)


def get_datasets(
    training_config: TrainingConfig,
    training_metadata: pd.DataFrame,
    training_path: str,
    validation_metadata: pd.DataFrame,
    validation_path: str,
) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    paths_list, labels_list = get_image_paths_and_labels(
        training_metadata, training_path, training_config
    )

    training_dataset = tf.data.Dataset.from_tensor_slices((paths_list, labels_list))
    training_dataset = (
        training_dataset.map(
            lambda path_tensor, label_tensor, data_w=training_config.input_w: tf.numpy_function(
                load_spectrogram_from_path,
                [path_tensor, label_tensor, data_w],
                (tf.float32, tf.int64),
            ),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        .map(_fixup_shape)
        .shuffle(buffer_size=512)
        .batch(training_config.batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    # Dump shapes of the dataset
    for images, labels in training_dataset.take(
        1
    ):  # only take first element of dataset
        numpy_images = images.numpy()
        numpy_labels = labels.numpy()

        print(f"Images shape: {numpy_images.shape}")
        print(f"Labels shape: {numpy_labels.shape}")

    # Dump training dataset
    # for batch_x, batch_y in dataset:
    #     print("*** BATCH ***")
    #     for i, (x, y) in enumerate(zip(batch_x, batch_y)):
    #         print(f"> EXAMPLE {i + 1}")
    #         print(f"text   : {x.numpy()} => shape: {x.shape}")
    #         print(f"target : {y.numpy()} => shape: {y.shape}")
    #         print()
    #     print()

    paths_list, labels_list = get_image_paths_and_labels(
        validation_metadata, validation_path, training_config
    )

    validation_dataset = tf.data.Dataset.from_tensor_slices((paths_list, labels_list))
    validation_dataset = (
        validation_dataset.map(
            lambda path_tensor, label_tensor, data_w=training_config.input_w: tf.numpy_function(
                load_spectrogram_from_path,
                [path_tensor, label_tensor, data_w],
                (tf.float32, tf.int64),
            ),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        .map(_fixup_shape)
        .shuffle(buffer_size=1024)
        .batch(training_config.batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    return training_dataset, validation_dataset
