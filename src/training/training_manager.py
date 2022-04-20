import pandas as pd
import tensorflow
import tensorflow as tf
from src.data_process.config_paths import DataPathsManager
from src.training.training_config import TrainingConfig
from src.data_process.metadata_processor import MetadataProcessor
import numpy as np
import os
from sklearn import preprocessing
from sklearn.utils import shuffle
from typing import Tuple
import gc


def load_data(
    path: str, metadata: pd.DataFrame, training_config: TrainingConfig
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
        try:
            # load the data
            data = np.load(os.path.join(path, f"{row[1]['track_id']}.npy"))
            # Check if the data is long enough
            if data.shape[1] > training_config.input_w:
                result.append(data)
                result_labels.append(row[1]["genre_top"])
        except FileNotFoundError:
            pass
        except ValueError:
            pass
    return result, result_labels


def prepare_data(
    training_config: TrainingConfig, org_data: list, labels: list
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare the data for the training. Each data is subarray of the spectogram of given length.
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
    training_config: TrainingConfig,
    metadata: pd.DataFrame,
    processor: MetadataProcessor,
    data_paths: DataPathsManager,
    SPLIT: int,
) -> None:

    label_encoder = preprocessing.LabelEncoder()

    training_config.model.compile(
        optimizer=training_config.optimizer,
        loss=training_config.loss,
        metrics=["accuracy"],
    )

    # TODO: Add data augmentation for split 2 and 3
    if SPLIT == 1:
        train, val, test = processor.split_metadata(metadata, 0.8, 0.1, 0.1)
    elif SPLIT == 2:
        train, val, test = processor.split_metadata_uniform(metadata, 0.8, 0.1, 0.1)
    elif SPLIT == 3:
        train, val, test = processor.split_metadata_uniform(
            metadata, 0.8, 0.1, 0.1, add_val_to_train=True
        )

    # Load all training and validation data into memory
    train_data, train_label = load_data(
        data_paths.get_train_dataset_path(SPLIT), train, training_config
    )
    val_data, val_label = load_data(
        data_paths.get_val_dataset_path(SPLIT), val, training_config
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

        # shuffle the data
        input_data, input_label = shuffle(input_data, input_label)
        val_input_data, val_input_label = shuffle(val_input_data, val_input_label)

        # train the model
        training_config.model.fit(
            input_data,
            input_label,
            epochs=1,
            validation_data=(val_input_data, val_input_label),
            batch_size=training_config.batch_size,
        )
        gc.collect()

    # Load test data
    test_data, test_label = load_data(
        data_paths.get_test_dataset_path(SPLIT), test, training_config
    )
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
    # Save model
    training_config.model.save(data_paths.get_model_path(SPLIT))
