import configparser


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
        self.datasetPath: str = (
            self.config.get("DATASET", "DATASET_PATH")
            + self.config.get("DATASET", "DATASET_PREFIX")
            + self.config.get("DATASET", "DATASET_NAME")
        )

        self.trainDatasetPath: str = self.datasetPath + self.config.get(
            "DATASET", "TRAIN_PATH"
        )
        self.valDatasetPath: str = self.datasetPath + self.config.get(
            "DATASET", "VAL_PATH"
        )
        self.testDatasetPath: str = self.datasetPath + self.config.get(
            "DATASET", "TEST_PATH"
        )
