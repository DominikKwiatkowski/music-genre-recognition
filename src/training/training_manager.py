import os
import shutil

import pandas as pd
import tensorflow as tf
import numpy as np
import gc
from typing import Tuple
from sklearn import preprocessing
from sklearn.utils import shuffle

from src.data_process.config_paths import DataPathsManager
from src.training.training_config import TrainingConfig


def load_data(
        path: str,
        metadata: pd.DataFrame,
        training_config: TrainingConfig
) -> Tuple[list, list]:
    """
    Load the data from the given path.
    :param path: Path to the data
    :param metadata: Metadata of the data
    :param training_config: Configuration for the training
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
        except (FileNotFoundError, ValueError):
            pass
    return result, result_labels


def prepare_data(
        training_config: TrainingConfig,
        org_data: list,
        labels: list
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
            data[:, starting_index: starting_index + training_config.input_w]
        )
    input_data = np.stack(input_data)
    input_label = np.array(labels)
    return input_data, input_label


def run_training(
        training_name: str,
        training_config: TrainingConfig,
        training_metadata: pd.DataFrame,
        validation_metadata: pd.DataFrame,
        data_paths: DataPathsManager,
        split_id: int,
        overwrite_previous: bool = False
) -> None:
    label_encoder = preprocessing.LabelEncoder()
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=f"{data_paths.training_log_path}{training_name}")

    # Protect previous models from overwriting
    if os.path.exists(f"{data_paths.model_path}{training_name}"):
        if overwrite_previous:
            print("WARNING: Model with the same name already exists. Overwriting it...")

            # Clear logs and model directories for previous run
            shutil.rmtree(f"{data_paths.training_log_path}{training_name}")
            shutil.rmtree(f"{data_paths.model_path}{training_name}")
        else:
            print("ERROR: Model with the same name already exists. Skipping...")
            print("INFO: To overwrite the model, use the overwrite_previous flag.")
            return

    # TODO: dump training config to file and save it to the "./logs/{training_name}"

    training_config.model.compile(
        optimizer=training_config.optimizer,
        loss=training_config.loss,
        metrics=["accuracy"],
    )

    # Load all training and validation data into memory
    train_data, train_label = load_data(
        data_paths.get_train_dataset_path(split_id), training_metadata, training_config
    )
    val_data, val_label = load_data(
        data_paths.get_val_dataset_path(split_id), validation_metadata, training_config
    )

    # Change labels from string to int
    train_label = label_encoder.fit_transform(train_label)
    val_label = label_encoder.fit_transform(val_label)

    # Every epoch has own data
    for _ in range(training_config.epochs):
        # Get subarrays for training and validation
        input_data, input_label = prepare_data(training_config, train_data, train_label)
        val_input_data, val_input_label = prepare_data(
            training_config, val_data, val_label
        )

        # Shuffle the data
        input_data, input_label = shuffle(input_data, input_label)
        val_input_data, val_input_label = shuffle(val_input_data, val_input_label)

        # Train the model
        training_config.model.fit(
            input_data,
            input_label,
            epochs=1,
            validation_data=(val_input_data, val_input_label),
            batch_size=training_config.batch_size,
            callbacks=tensorboard_callback,
        )

        # Collect garbage to avoid memory leak
        gc.collect()

    # # Load test data
    # test_data, test_label = load_data(
    #     data_paths.get_test_dataset_path(split_id), test, training_config
    # )
    # test_label = label_encoder.fit_transform(test_label)
    # test_input_data, test_input_label = prepare_data(
    #     training_config, test_data, test_label
    # )
    #
    # # Shuffle and run test data
    # test_input_data, test_input_label = shuffle(test_input_data, test_input_label)
    # test_loss, test_acc = training_config.model.evaluate(
    #     test_input_data, test_input_label
    # )
    #
    # # Print result
    # print(f"Test loss: {test_loss}, test accuracy: {test_acc}")

    # Save model
    training_config.model.save(data_paths.model_path + training_name)

    # Clear gpu session
    tf.keras.backend.clear_session()
