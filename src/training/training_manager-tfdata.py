import os
import shutil
import pandas as pd
import tensorflow as tf
import gc
from sklearn import preprocessing

from src.data_process.config_paths import DataPathsManager
from src.training.training_config import TrainingSetup, TrainingParams
from training.training_data_generator import get_datasets

label_encoder = preprocessing.LabelEncoder()


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


def run_training_new(
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
    :param data_paths: Paths to the data
    :param augment: Augment the data
    :param overwrite_previous: Overwrite previous training
    :return:
    """
    training_config = TrainingSetup(TrainingParams())

    training_dataset, validation_dataset = get_datasets(
        training_config,
        training_metadata,
        training_path,
        validation_metadata,
        validation_path,
    )

    # Setup callbacks
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=f"{data_paths.training_log_path}{training_name}", update_freq="epoch"
    )

    # Learning rate tweaker which decreases the learning rate if loss is not decreasing
    lr_tweaker = tf.keras.callbacks.ReduceLROnPlateau(
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

    # TODO: dump training config to file and save it to the "./logs/{training_name}"

    training_config.model.compile(
        optimizer=training_config.optimizer,
        loss=training_config.loss,
        metrics=["accuracy"],
    )

    training_config.model.fit(
        training_dataset,
        epochs=training_config.p.epochs,
        batch_size=training_config.p.batch_size,
        validation_data=validation_dataset,
        shuffle=True,
        callbacks=[tensorboard_callback, lr_tweaker],
    )

    # Clear gpu session
    tf.keras.backend.clear_session()

    # Collect garbage to avoid memory leak
    gc.collect()
