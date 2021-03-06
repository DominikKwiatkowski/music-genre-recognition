import os
import shutil
import math
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import numpy as np
import gc
from typing import Tuple
from sklearn import preprocessing
from sklearn.utils import shuffle
import datetime
import time

from src.data_process.config_paths import DataPathsManager
from src.training.training_config import TrainingSetup, TrainingParams
from src.data_process.spectrogram_augmenter import noise_overlay, mask_spectrogram
from src.training.LrTweaker import LrTweaker


def calculate_eta(curr_step: int, total_steps: int, start_time: float) -> str:
    """
    Calculate the time left for the process.
    :param curr_step: Current step
    :param total_steps: Total steps
    :param start_time: Start time
    :return: Time left for the training as string.
    """

    if curr_step == 0:
        return "ETA:  --:--:--"

    time_left = (total_steps - curr_step) * (time.time() - start_time) / curr_step
    return str(datetime.timedelta(seconds=time_left))


def prepare_output_dirs(
    model_path: str,
    training_log_path: str,
    training_name: str,
    overwrite_previous: bool,
) -> None:
    """
    Prepare the output directories for the training.
    :param model_path: Path to the model
    :param training_log_path: Path to the training log
    :param training_name: Name of the training
    :param overwrite_previous: Overwrite previous training
    :return:
    """

    if os.path.exists(f"{model_path}{training_name}"):
        if overwrite_previous:
            print("WARNING: Model with the same name already exists. Overwriting it...")
            shutil.rmtree(f"{model_path}{training_name}")
        else:
            print("ERROR: Model with the same name already exists. Skipping...")
            print("INFO: To overwrite the models, use the overwrite_previous flag.")
            return

    if os.path.exists(f"{training_log_path}{training_name}"):
        if overwrite_previous:
            print(
                "WARNING: Logs with the same name already exists. Overwriting them..."
            )
            shutil.rmtree(f"{training_log_path}{training_name}")
        else:
            print("ERROR: Logs with the same name already exists. Skipping...")
            print("INFO: To overwrite the logs, use the overwrite_previous flag.")
            return


def augment_data(spectrogram: np.ndarray) -> list:
    """
    Augment the given data.
    :param spectrogram: Data to augment
    :return: Augmented data
    """
    result = [
        mask_spectrogram(spectrogram, n_freq_masks=1, n_time_masks=0),
        # mask_spectrogram(spectrogram, n_freq_masks=3, n_time_masks=0),
        noise_overlay(spectrogram, noise_pct=0.7, noise_amt=0.05),
    ]

    return result


def load_data(
    training_config: TrainingSetup,
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
            if data.shape[1] > training_config.p.input_w:
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
    training_config: TrainingSetup, org_data: list, labels: list
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
        starting_index = np.random.randint(0, data.shape[1] - training_config.p.input_w)
        input_data.append(
            data[:, starting_index : starting_index + training_config.p.input_w]
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
    test_metadata: pd.DataFrame,
    test_path: str,
    data_paths: DataPathsManager,
    augment: bool,
    overwrite_previous: bool = False,
) -> None:
    """
    Run the training.
    :param training_name: Name of the training
    :param training_metadata: Metadata of the training data
    :param training_path: Path to the training data
    :param validation_metadata: Metadata of the validation data
    :param validation_path: Path to the validation data
    :param test_metadata: Metadata of the test data
    :param test_path: Path to the test data
    :param data_paths: Paths to the data
    :param augment: Augment the data
    :param overwrite_previous: Overwrite previous training
    :return:
    """
    training_config = TrainingSetup(TrainingParams())

    # Setup callbacks
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=f"{data_paths.training_log_path}{training_name}", update_freq="epoch"
    )

    # Learning rate tweaker which decreases the learning rate if loss is not decreasing
    lr_tweaker = LrTweaker(
        training_config,
        patience=training_config.p.learning_rate_patience,
        decrease_multiplier=training_config.p.learning_rate_decrease_multiplier,
        min_lr=training_config.p.learning_rate_min,
    )

    # Dummy ReduceLROnPlateau which is bugged and doesn't work, but is good to display learning rate with verbose
    dummy_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="loss",
        factor=training_config.p.learning_rate_decrease_multiplier,
        patience=training_config.p.learning_rate_patience,
        min_lr=training_config.p.learning_rate_min,
    )

    prepare_output_dirs(
        data_paths.training_model_path,
        data_paths.training_log_path,
        training_name,
        overwrite_previous,
    )

    # Save initial checkpoint of training
    training_config.save(training_name, "init", data_paths.training_model_path)

    training_config.model.compile(
        optimizer=training_config.optimizer,
        loss=training_config.loss,
        metrics=["accuracy"],
    )

    # Load all training and validation data into memory
    train_data, train_label = load_data(
        training_config, training_metadata, training_path, augment
    )
    val_data, val_label = load_data(
        training_config, validation_metadata, validation_path
    )

    # Change labels from string to int
    train_label = training_config.label_encoder.fit_transform(train_label)
    val_label = training_config.label_encoder.fit_transform(val_label)

    # Epoch ETA estimator
    training_start_time = time.time()

    # Collect garbage to avoid memory leak
    gc.collect()

    # Every epoch has own data
    for epoch_id in range(training_config.p.starting_epoch, training_config.p.epochs):
        eta = calculate_eta(epoch_id, training_config.p.epochs, training_start_time)
        print(f"Epoch: {epoch_id}/{training_config.p.epochs}. ETA: {eta}")

        # Get subarrays for training and validation
        input_data, input_label = prepare_data(training_config, train_data, train_label)
        val_input_data, val_input_label = prepare_data(
            training_config, val_data, val_label
        )

        # Shuffle the data
        input_data, input_label = shuffle(input_data, input_label)
        val_input_data, val_input_label = shuffle(val_input_data, val_input_label)

        # Split data to parts of equal size
        if input_data.shape[0] > training_config.p.patch_size:
            input_data = np.array_split(
                input_data,
                math.ceil(input_data.shape[0] / training_config.p.patch_size),
            )
            input_label = np.array_split(
                input_label,
                math.ceil(input_label.shape[0] / training_config.p.patch_size),
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
                initial_epoch=epoch_id,
                epochs=epoch_id + 1,
                batch_size=training_config.p.batch_size,
                validation_data=(val_input_data, val_input_label),
                shuffle=True,
                callbacks=[tensorboard_callback, dummy_lr],
            )

            # Tweak model's learning rate
            lr_tweaker.on_epoch_end(epoch_story.history["loss"][0])

            # Clear gpu session
            tf.keras.backend.clear_session()

            # Collect garbage to avoid memory leak
            gc.collect()

        # Save models after each epoch
        training_config.save(
            training_name, f"{epoch_id}", data_paths.training_model_path
        )

        # Test model on training data(debug only)
        # tm.test_model_training(
        #     training_name,
        #     training_config,
        #     data_paths,
        #     test_metadata,
        #     test_path,
        #     epoch_id,
        # )
