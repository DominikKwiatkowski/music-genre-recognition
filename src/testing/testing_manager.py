import os
import shutil
import math
import pandas as pd
import src.training.training_manager as tm
import tensorflow as tf
from tensorflow.keras import backend as K
import seaborn as sns
import numpy as np
import io
import gc
from typing import Tuple
from sklearn import preprocessing
from sklearn.utils import shuffle
from tensorflow import keras
from src.data_process.config_paths import DataPathsManager
from src.training.training_config import TrainingConfig
from matplotlib import pyplot as plt


def log_confusion_matrix(
    training_config: TrainingConfig,
    test_data: np.ndarray,
    test_labels: np.ndarray,
    training_name: str,
    data_paths: DataPathsManager,
    step: int,
) -> None:
    # Use the model to predict the values from the validation dataset.
    test_pred = training_config.model.predict(test_data)
    test_pred_class = np.argmax(test_pred, axis=1)
    print(test_pred_class)
    con_mat = tf.math.confusion_matrix(
        labels=test_labels, predictions=test_pred_class
    ).numpy()
    con_mat_norm = np.around(
        con_mat.astype("float") / con_mat.sum(axis=1)[:, np.newaxis], decimals=2
    )

    con_mat_df = pd.DataFrame(con_mat_norm)

    figure = plt.figure(figsize=(8, 8))
    sns.heatmap(con_mat_df, annot=True, cmap=plt.cm.Blues)
    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")

    buf = io.BytesIO()
    plt.savefig(buf, format="png")

    plt.close(figure)
    buf.seek(0)
    image = tf.image.decode_png(buf.getvalue(), channels=4)

    image = tf.expand_dims(image, 0)

    # Log the confusion matrix as an image summary.
    w = tf.summary.create_file_writer(
        logdir=f"{data_paths.training_log_path}{training_name}"
    )
    with w.as_default():
        tf.summary.image("Confusion Matrix" + str(step), image, step=step)
    gc.collect()


def test_model_training(
    training_name: str,
    training_config: TrainingConfig,
    data_paths: DataPathsManager,
    test_metadata: pd.DataFrame,
    test_path: str,
    step: int,
) -> None:
    # training_config = TrainingConfig()

    # training_config.model = keras.models.load_model(data_paths.model_path + model_name)
    # Load test data
    test_data, test_label = tm.load_data(training_config, test_metadata, test_path)
    test_label = tm.label_encoder.fit_transform(test_label)
    test_input_data, test_input_label = tm.prepare_data(
        training_config, test_data, test_label
    )

    # Shuffle and run test data
    test_input_data, test_input_label = shuffle(test_input_data, test_input_label)
    log_confusion_matrix(
        training_config,
        test_input_data,
        test_input_label,
        training_name,
        data_paths,
        step,
    )


def test_model(
    model_name: str,
    model_id: str,
    data_paths: DataPathsManager,
    test_metadata: pd.DataFrame,
    test_path: str,
    step: int,
) -> None:
    training_config = TrainingConfig()
    encoder = preprocessing.LabelEncoder()
    training_config.model.load_weights(
        data_paths.model_path + model_name + model_id + ".h5"
    )
    encoder.classes_ = np.load(data_paths.model_path + model_name + "label_encoder.npy")
    training_config.model.summary()
    # Load test data
    test_data, test_label = tm.load_data(training_config, test_metadata, test_path)
    test_label = tm.label_encoder.fit_transform(test_label)
    test_input_data, test_input_label = tm.prepare_data(
        training_config, test_data, test_label
    )

    # Shuffle and run test data
    test_input_data, test_input_label = shuffle(test_input_data, test_input_label)
    log_confusion_matrix(
        training_config,
        test_input_data,
        test_input_label,
        "test_mode" + model_name,
        data_paths,
        step,
    )
