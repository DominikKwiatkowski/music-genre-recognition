import os
import shutil
import math
import pandas as pd
import tensorflow as tf
from tensorflow.keras import backend as K
import numpy as np
import gc
from typing import Tuple
from sklearn import preprocessing
from sklearn.utils import shuffle

from src.data_process.config_paths import DataPathsManager
from src.training.training_config import TrainingConfig
from src.data_process.spectrogram_augmenter import noise_overlay, mask_spectrogram

label_encoder = preprocessing.LabelEncoder()


def augment_data(spectrogram: np.ndarray) -> list:
    """
    Augment the given data.
    :param spectrogram: Data to augment
    :return: Augmented data
    """
    result = [
        mask_spectrogram(spectrogram, n_freq_masks=1, n_time_masks=1),
        noise_overlay(spectrogram),
    ]

    return result


def load_data(
    training_config: TrainingConfig,
    metadata: pd.DataFrame,
    path: str,
    augment: bool = False,
) -> Tuple[list, list]:
    """
    Load the data from the given path.
    :param training_config: Configuration for the training
    :param metadata: Metadata of the data
    :param path: Path to the data
    :param augment: Augment loaded data
    :return:
    """

    result = []
    result_labels = []
    for row in metadata.iterrows():
        # TODO: Move validation of data to separate module
        try:
            # Load the data
            data = np.load(os.path.join(path, f"{row[1]['track_id']}.npy"))
            # Check if the data is long enough
            if data.shape[1] > training_config.input_w:
                result.append(data)
                result_labels.append(row[1]["genre_top"])
                if augment:
                    aug_data = augment_data(data)
                    result.extend(aug_data)
                    # Add the same label for each augmentation
                    result_labels.extend([row[1]["genre_top"]] * len(aug_data))
        except (FileNotFoundError, ValueError):
            pass

    return result, result_labels


def prepare_data(
    training_config: TrainingConfig, org_data: list, labels: list
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare the data for the training. Each data is subarray of the spectrogram of given length.
    :param training_config: Configuration for the training
    :param org_data: Original data
    :param labels: Labels for the data
    :return:
    """
    input_data = []
    for data in org_data:
        # Find starting point for each data
        starting_index = np.random.randint(0, data.shape[1] - training_config.input_w)
        input_data.append(
            data[:, starting_index : starting_index + training_config.input_w]
        )
    input_data = np.stack(input_data)
    input_label = np.array(labels)

    return input_data, input_label


def run_training(
    training_name: str,
    training_metadata: pd.DataFrame,
    training_path: str,
    validation_metadata: pd.DataFrame,
    validation_path: str,
    data_paths: DataPathsManager,
    augment: bool,
    overwrite_previous: bool = False,
) -> None:
    training_config = TrainingConfig()
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=f"{data_paths.training_log_path}{training_name}",
    )

    # Protect previous models from overwriting
    if os.path.exists(f"{data_paths.model_path}{training_name}"):
        if overwrite_previous:
            print("WARNING: Model with the same name already exists. Overwriting it...")
            shutil.rmtree(f"{data_paths.model_path}{training_name}")
        else:
            print("ERROR: Model with the same name already exists. Skipping...")
            print("INFO: To overwrite the models, use the overwrite_previous flag.")

    if os.path.exists(f"{data_paths.training_log_path}{training_name}"):
        if overwrite_previous:
            print(
                "WARNING: Logs with the same name already exists. Overwriting them..."
            )
            shutil.rmtree(f"{data_paths.training_log_path}{training_name}")
        else:
            print("ERROR: Logs with the same name already exists. Skipping...")
            print("INFO: To overwrite the logs, use the overwrite_previous flag.")

    # TODO: dump training config to file and save it to the "./logs/{training_name}"

    training_config.model.compile(
        optimizer=training_config.optimizer,
        loss=training_config.loss,
        metrics=["accuracy"],
    )

    best_loss: float = np.inf
    not_improved_count: int = 0

    # Load all training and validation data into memory
    train_data, train_label = load_data(
        training_config, training_metadata, training_path, augment
    )
    val_data, val_label = load_data(
        training_config, validation_metadata, validation_path
    )

    # Change labels from string to int
    train_label = label_encoder.fit_transform(train_label)
    val_label = label_encoder.fit_transform(val_label)

    # Every epoch has own data
    for epoch_id in range(training_config.epochs):
        print(f"Epoch: {epoch_id}")

        # Get subarrays for training and validation
        input_data, input_label = prepare_data(training_config, train_data, train_label)
        val_input_data, val_input_label = prepare_data(
            training_config, val_data, val_label
        )

        # Shuffle the data
        input_data, input_label = shuffle(input_data, input_label)
        val_input_data, val_input_label = shuffle(val_input_data, val_input_label)

        # Split data to parts of size 6400
        # TODO: Find better solution
        if input_data.shape[0] > training_config.patch_size:
            input_data = np.array_split(
                input_data, math.ceil(input_data.shape[0] / training_config.patch_size)
            )
            input_label = np.array_split(
                input_label,
                math.ceil(input_label.shape[0] / training_config.patch_size),
            )
        else:
            input_data = [input_data]
            input_label = [input_label]

        fits_per_epoch: int = len(input_data)

        # For each part of data, run models training
        for i in range(fits_per_epoch):
            epoch_story = training_config.model.fit(
                input_data[i],
                input_label[i],
                epochs=1,
                batch_size=training_config.batch_size,
                validation_data=(val_input_data, val_input_label),
                shuffle=True,
                callbacks=tensorboard_callback,
            )

            # Clear gpu session
            gc.collect()

            # Collect garbage to avoid memory leak
            gc.collect()

        # Save models after each epoch
        training_config.model.save(data_paths.model_path + training_name)

        # Update best loss
        best_loss = min(best_loss, epoch_story.history["loss"][0])

        # If loss is not decreasing since last 3 epochs, reduce learning rate 0.5 times
        if epoch_story.history["loss"][0] > best_loss:
            not_improved_count += 1

            if not_improved_count >= training_config.learning_rate_patience:
                old_lr = training_config.learning_rate
                training_config.learning_rate *= 0.5
                not_improved_count = 0

                # Update new learning rate to the optimizer
                K.set_value(
                    training_config.model.optimizer.learning_rate,
                    training_config.learning_rate,
                )
                print(
                    f"Epoch: {epoch_id} Reducing lr from {old_lr} to {training_config.learning_rate}"
                )


def test_model(
    training_config: TrainingConfig, test_metadata: pd.DataFrame, test_path: str
) -> None:
    # Load test data
    test_data, test_label = load_data(training_config, test_metadata, test_path)
    test_label = label_encoder.fit_transform(test_label)
    test_input_data, test_input_label = prepare_data(
        training_config, test_data, test_label
    )

    # Shuffle and run test data
    test_input_data, test_input_label = shuffle(test_input_data, test_input_label)
    test_loss, test_acc = training_config.model.evaluate(
        test_input_data, test_input_label
    )
    # Print result
    print(f"Test loss: {test_loss}, test accuracy: {test_acc}")
