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


def load_data(
    path: str, metadata: pd.DataFrame, training_config: TrainingConfig
) -> Tuple[list, list]:
    # Load all npy files in the given path and append them to a result array
    result = []
    result_labels = []
    for row in metadata.iterrows():
        try:
            data = np.load(os.path.join(path, f"{row[1]['track_id']}.npy"))
            if data.shape[1] > training_config.input_w:
                result.append(data)
                result_labels.append(row[1]["genre_top"])
        except FileNotFoundError:
            pass
    return result, result_labels


def prepare_data(
    training_config: TrainingConfig, org_data: list, labels: list
) -> np.ndarray:
    input_data = []
    for data in org_data:
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
) -> None:
    label_encoder = preprocessing.LabelEncoder()
    training_config.model.compile(
        optimizer=training_config.optimizer,
        loss=training_config.loss,
        metrics=["accuracy"],
    )
    # TODO: Add some runner to run all splits training at once
    SPLIT = 2
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
        data_paths.trainDatasetPath, train, training_config
    )
    val_data, val_label = load_data(data_paths.valDatasetPath, val, training_config)
    # Change labels from string to int
    train_label = label_encoder.fit_transform(train_label)
    val_label = label_encoder.fit_transform(val_label)
    for _ in range(training_config.epochs):
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

    # Evaluate the model
    test_data, test_label = load_data(data_paths.testDatasetPath, test, training_config)
    test_label = label_encoder.fit_transform(test_label)
    test_input_data, test_input_label = prepare_data(
        training_config, test_data, test_label
    )
    test_input_data, test_input_label = shuffle(test_input_data, test_input_label)
    test_loss, test_acc = training_config.model.evaluate(
        test_input_data, test_input_label
    )
    print(f"Test loss: {test_loss}, test accuracy: {test_acc}")
