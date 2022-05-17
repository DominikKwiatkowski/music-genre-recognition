import configparser
import os


class DataPathsManager:
    def __init__(self):
        self.config = configparser.ConfigParser()
        self.config.read("config.ini")

        # Generate paths for data set
        self.subsetFilePath: str = self.config.get(
            "METADATA", "METADATA_PATH"
        ) + self.config.get("METADATA", "SUBSET_FILE")
        self.labelFilePath: str = self.config.get(
            "METADATA", "METADATA_PATH"
        ) + self.config.get("METADATA", "LABEL_FILE")

        self.datasetName: str = self.config.get("DATASET", "DATASET_NAME")
        self.datasetDir: str = self.config.get("DATASET", "DATASET_PATH")
        self.datasetPath: str = (
            self.datasetDir
            + self.config.get("DATASET", "DATASET_PREFIX")
            + self.config.get("DATASET", "DATASET_NAME")
        )

        # Load prepared data paths
        self.trainDatasetPath: str = self.datasetDir + self.config.get(
            "DATASET", "PREP_TRAIN_PATH"
        )
        self.valDatasetPath: str = self.datasetDir + self.config.get(
            "DATASET", "PREP_VAL_PATH"
        )
        self.testDatasetPath: str = self.datasetDir + self.config.get(
            "DATASET", "PREP_TEST_PATH"
        )

        # Load and initialize training paths
        self.training_root_path: str = self.config.get("TRAINING", "TRAINING_ROOT_PATH")
        self.training_log_path: str = self.config.get("TRAINING", "TRAINING_LOG_PATH")
        self.training_model_path: str = self.config.get("TRAINING", "MODEL_PATH")

        if not os.path.exists(self.training_root_path):
            os.mkdir(self.training_root_path)
        if not os.path.exists(self.training_log_path):
            os.mkdir(self.training_log_path)
        if not os.path.exists(self.training_model_path):
            os.mkdir(self.training_model_path)

    def get_train_dataset_path(self, split: int) -> str:
        return f"{self.trainDatasetPath}{split}/"

    def get_val_dataset_path(self, split: int) -> str:
        return f"{self.valDatasetPath}{split}/"

    def get_test_dataset_path(self, split: int) -> str:
        return f"{self.testDatasetPath}{split}/"
