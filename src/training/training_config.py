import json
import os
import shutil

import numpy as np
import tensorflow as tf
from tensorflow.keras import mixed_precision
from sklearn import preprocessing

from models.multi_scale_level_cnn import multi_scale_level_cnn
from models.dumb import dumb_model

# ============================================
# Setup default Tensorflow parameters
# ============================================
gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# Enable eager execution for debugging
# tf.compat.v1.enable_eager_execution()

# Equivalent to the two lines above
mixed_precision.set_global_policy("mixed_float16")

# ============================================
# All non-immediate variables in TrainingConfig must be defined as keys of the dictionary
# ============================================
optimizers = {"Adam": tf.keras.optimizers.Adam()}

losses = {
    "SparseCategoricalCrossentropy": tf.keras.losses.SparseCategoricalCrossentropy()
}

initializers = {
    "RandomNormal": tf.keras.initializers.RandomNormal(
        mean=0.0, stddev=0.05, seed=None
    ),
    "VarianceScaling": tf.keras.initializers.VarianceScaling(
        scale=1.0, mode="fan_in", distribution="truncated_normal", seed=None
    ),
}


class TrainingParams:
    """
    TrainingParams is an object that contains all the training parameters.
    It should not contain any complex types, as it should be serializable.
    """

    def __init__(self):
        # Output classes
        self.num_classes: int = 8

        # Batch size for training
        self.batch_size: int = 8

        # Number of epochs to train for
        self.starting_epoch: int = 0
        self.epochs: int = 100

        # Learning rate
        self.learning_rate: float = 0.01
        self.learning_rate_patience: int = 3
        self.learning_rate_decrease_multiplier: float = 0.5
        self.learning_rate_min: float = 0.0

        # Early stopping conditions
        self.early_stopping_patience: int = 10
        self.early_stopping_min_delta: float = 0.01

        # Input shape
        self.input_h = 128  # Always 128
        self.input_w = 512  # Corresponds to the track's length; 512 is around 6 seconds
        self.patch_size = 3276800 / self.input_w  # size which can be allocated on GPU

        # Optimizer
        self.optimizer_name: str = "Adam"

        # Loss
        self.loss_name: str = "SparseCategoricalCrossentropy"

        # Initializer
        self.initializer_name: str = "RandomNormal"

        # Model name
        self.model_name: str = "dumb_model"


class TrainingSetup:
    """
    TrainingSetup is an object that contains all the training objects.
    It is referenced during the training process.
    """

    def __init__(self, params: TrainingParams = TrainingParams()):
        # Copy all params to TrainingSetup
        self.p: TrainingParams = params

        self.optimizer: tf.keras.Optimizer = None
        self.loss: tf.keras.loss = None
        self.initializer: tf.keras.initializers = None
        self.label_encoder = preprocessing.LabelEncoder()

        if params is None:
            # If no params are passed, do not initialize config
            # To save time on model initialization
            print("WARNING: TrainingSetup not initialized.")
            return

        self.init_optimizer()
        self.init_loss()
        self.init_initializer()

        # Initialize Model layers
        if self.p.model_name == "multi_scale_level_cnn":
            self.model = multi_scale_level_cnn(
                (self.p.input_h, self.p.input_w, 1),
                num_dense_blocks=3,
                num_conv_filters=32,
                num_classes=self.p.num_classes,
                initializer=self.initializer,
            )
        elif self.p.model_name == "dumb_model":
            self.model = dumb_model(
                (self.p.input_h, self.p.input_w, 1),
                num_classes=self.p.num_classes,
                initializer=self.initializer,
            )
        else:
            raise ValueError("Model name not recognized")

        self.model.summary()

    def init_optimizer(self):
        self.optimizer: tf.keras.Optimizer = optimizers[self.p.optimizer_name]
        self.optimizer.learning_rate = self.p.learning_rate

    def init_loss(self):
        self.loss: tf.keras.loss = losses[self.p.loss_name]

    def init_initializer(self):
        self.initializer: tf.keras.initializers.Initializer = initializers[
            self.p.initializer_name
        ]

    # ============================================
    # TrainingSetup serialization and deserialization
    # ============================================
    def save(self, name: str, capture_id: str = "0", path: str = ".") -> None:
        """
        Save the TrainingSetup object to a file
        :param name: name of the file to save to
        :param capture_id: ID of the capture
        :param path: path to save the file to
        """

        # Paths
        root_dir: str = os.path.join(path, name)
        encoder_path: str = os.path.join(root_dir, "label_encoder.npy")
        model_path: str = os.path.join(root_dir, "model_architecture.json")
        captures_dir: str = os.path.join(root_dir, "captures", f"{capture_id}")
        params_path: str = os.path.join(captures_dir, "params.json")
        weight_path: str = os.path.join(captures_dir, "weights.h5")

        # Validation to avoid overwriting and corrupting models
        if os.path.exists(model_path):
            if os.path.exists(captures_dir):
                print(
                    f"WARNING: Checkpoint for epoch {capture_id} is already saved. Overwriting..."
                )
                shutil.rmtree(captures_dir)
        else:
            # Make sure directory is clean
            shutil.rmtree(root_dir, ignore_errors=True)

        # Prepare directory for setup save
        if not os.path.exists(root_dir):
            os.makedirs(root_dir)

        # Serialize Model architecture
        if not os.path.exists(model_path):
            with open(model_path, "w") as f:
                f.write(self.model.to_json())

        # Serialize Weights
        if not os.path.exists(captures_dir):
            os.makedirs(captures_dir)

        # Serialize TrainingParams
        if not os.path.exists(params_path):
            with open(params_path, "w") as f:
                params = vars(self.p)
                f.write(json.dumps(params))

        # Serialize label encoder
        if hasattr(self.label_encoder, "classes_"):
            np.save(
                encoder_path,
                self.label_encoder.classes_,
            )

        self.model.save_weights(weight_path)

    def load(self, name: str, capture_id: str = "0", path: str = ".") -> None:
        """
        Load the TrainingSetup object from a file
        :param name: name of the file to load from
        :param capture_id: ID of the capture
        :param path: path to load the file from
        """

        # Paths
        root_dir: str = os.path.join(path, name)
        encoder_path: str = os.path.join(root_dir, "label_encoder.npy")
        model_path: str = os.path.join(root_dir, "model_architecture.json")
        captures_dir: str = os.path.join(root_dir, "captures", f"{capture_id}")
        params_path: str = os.path.join(captures_dir, "params.json")
        weight_path: str = os.path.join(captures_dir, "weights.h5")

        # Load Model architecture
        if os.path.exists(model_path):
            with open(model_path) as f:
                self.model = tf.keras.models.model_from_json(f.read())
        else:
            raise ValueError(f"No model file found at {model_path}")

        # Load TrainingParams
        if os.path.exists(params_path):
            with open(params_path) as f:
                params = json.loads(f.read())

                for key, value in params.items():
                    setattr(self.p, key, value)
        else:
            raise ValueError(f"No params file found at {params_path}")

        # Load label encoder
        if os.path.exists(encoder_path):
            setattr(self.label_encoder, "classes_", np.load(encoder_path))
        else:
            print(f"No encoder file found at {encoder_path}")

        # Load Weights
        if os.path.exists(captures_dir):
            self.model.load_weights(weight_path)
        else:
            raise ValueError(f"No weights file found at {weight_path}")

        self.init_optimizer()
        self.init_loss()
        self.init_initializer()
