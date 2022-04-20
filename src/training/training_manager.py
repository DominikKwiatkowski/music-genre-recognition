import math

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
from src.data_process.spectrogram_augmenter import noise_overlay, mask_spectrogram


class TrainingManager:
    def __init__(self, training_config: TrainingConfig):
        self.training_config = training_config
        self.label_encoder = preprocessing.LabelEncoder()

    def load_data(
        self, path: str, metadata: pd.DataFrame, augment: bool = False
    ) -> Tuple[list, list]:
        """
        Load the data from the given path.
        :param path: Path to the data
        :param metadata: Metadata of the data
        :return:
        """
        result = []
        result_labels = []
        for row in metadata.iterrows():
            try:
                # load the data
                data = np.load(os.path.join(path, f"{row[1]['track_id']}.npy"))
                # Check if the data is long enough
                if data.shape[1] > self.training_config.input_w:
                    result.append(data)
                    result_labels.append(row[1]["genre_top"])
                    if augment:
                        aug_data = self.augment(data)
                        result.extend(aug_data)
                        # Add the same label for each augmentation
                        result_labels.extend([row[1]["genre_top"]] * len(aug_data))
            except FileNotFoundError:
                pass
            except ValueError:
                pass
        return result, result_labels

    def prepare_data(
        self, org_data: list, labels: list
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare the data for the training. Each data is subarray of the spectogram of given length.
        :param org_data: Original data
        :param labels: Labels for the data
        :return:
        """
        input_data = []
        for data in org_data:
            # Find starting point for each data
            starting_index = np.random.randint(
                0, data.shape[1] - self.training_config.input_w
            )
            input_data.append(
                data[:, starting_index : starting_index + self.training_config.input_w]
            )
        input_data = np.stack(input_data)
        input_label = np.array(labels)
        return input_data, input_label

    @staticmethod
    def augment(spectrogram: np.ndarray) -> list:
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

    def run_training(
        self,
        train: pd.DataFrame,
        train_path: str,
        val: pd.DataFrame,
        val_path: str,
        augment: bool,
    ) -> None:
        self.training_config.model.compile(
            optimizer=self.training_config.optimizer,
            loss=self.training_config.loss,
            metrics=["accuracy"],
        )

        # Load all training and validation data into memory
        train_data, train_label = self.load_data(train_path, train, augment)
        val_data, val_label = self.load_data(val_path, val)

        # Change labels from string to int
        train_label = self.label_encoder.fit_transform(train_label)
        val_label = self.label_encoder.fit_transform(val_label)

        # Every epoch has own data
        for _ in range(self.training_config.epochs):
            # Get subarrays for training and validation
            input_data, input_label = self.prepare_data(train_data, train_label)
            val_input_data, val_input_label = self.prepare_data(val_data, val_label)

            # Shuffle the data
            input_data, input_label = shuffle(input_data, input_label)
            val_input_data, val_input_label = shuffle(val_input_data, val_input_label)
            # split data to parts of size 6400
            # TODO: Find better solution
            if input_data.shape[0] > 6400:
                input_data = np.array_split(
                    input_data, math.ceil(input_data.shape[0] / 6400)
                )
                input_label = np.array_split(
                    input_label, math.ceil(input_label.shape[0] / 6400)
                )
            else:
                input_data = [input_data]
                input_label = [input_label]

            # For each part of data, run epoch.
            for i in range(len(input_data)):
                self.training_config.model.fit(
                    input_data[i],
                    input_label[i],
                    epochs=1,
                    batch_size=self.training_config.batch_size,
                    validation_data=(val_input_data, val_input_label),
                    shuffle=True,
                )
                gc.collect()

    def test_model(self, test: pd.DataFrame, test_path: str) -> None:
        # Load test data
        test_data, test_label = self.load_data(test_path, test)
        test_label = self.label_encoder.fit_transform(test_label)
        test_input_data, test_input_label = self.prepare_data(test_data, test_label)

        # Shuffle and run test data
        test_input_data, test_input_label = shuffle(test_input_data, test_input_label)
        test_loss, test_acc = self.training_config.model.evaluate(
            test_input_data, test_input_label
        )
        # Print result
        print(f"Test loss: {test_loss}, test accuracy: {test_acc}")
