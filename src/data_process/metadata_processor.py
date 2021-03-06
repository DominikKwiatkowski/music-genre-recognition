import ast
import os
from typing import Tuple
import matplotlib.pyplot as plt
import pandas as pd

from src.data_process.config_paths import DataPathsManager


class MetadataProcessor:
    def __init__(self):
        self.data_paths = DataPathsManager()

    def load_full_metadata(self) -> pd.DataFrame:
        """
        Loads the label file. Source: https://github.com/mdeff/fma/blob/master/utils.py
        :return: DataFrame with the trackMetadata
        """
        trackMetadata = pd.read_csv(
            self.data_paths.labelFilePath, index_col=0, header=[0, 1]
        )
        COLUMNS = [
            ("track", "tags"),
            ("album", "tags"),
            ("artist", "tags"),
            ("track", "genres"),
            ("track", "genres_all"),
        ]
        for column in COLUMNS:
            trackMetadata[column] = trackMetadata[column].map(ast.literal_eval)

        COLUMNS = [
            ("track", "date_created"),
            ("track", "date_recorded"),
            ("album", "date_created"),
            ("album", "date_released"),
            ("artist", "date_created"),
            ("artist", "active_year_begin"),
            ("artist", "active_year_end"),
        ]
        for column in COLUMNS:
            trackMetadata[column] = pd.to_datetime(trackMetadata[column])

        SUBSETS = ("small", "medium", "large")
        try:
            trackMetadata["set", "subset"] = trackMetadata["set", "subset"].astype(
                "category", categories=SUBSETS, ordered=True
            )
        except (ValueError, TypeError):
            # the categories and ordered arguments were removed in pandas 0.25
            trackMetadata["set", "subset"] = trackMetadata["set", "subset"].astype(
                pd.CategoricalDtype(categories=SUBSETS, ordered=True)
            )

        COLUMNS = [
            ("track", "genre_top"),
            ("track", "license"),
            ("album", "type"),
            ("album", "information"),
            ("artist", "bio"),
        ]
        for column in COLUMNS:
            trackMetadata[column] = trackMetadata[column].astype("category")

        return trackMetadata

    def generate_metadata_csv(self) -> pd.DataFrame:
        """
        Creates a csv file with the data set. Only track id and genre are saved.
        :return:
        """
        # Read label file, which is csv file
        trackMetadata = self.load_full_metadata()

        # Get dataset specific metadata for tracks
        subsetMetadata = trackMetadata[
            trackMetadata["set", "subset"] <= self.data_paths.datasetName
        ]

        # Change metadata scope to genre information only
        subsetMetadata = subsetMetadata["track"]["genre_top"]

        # Save genres to csv
        subsetMetadata.to_csv(self.data_paths.subsetFilePath)

        return subsetMetadata

    def get_metadata(self) -> pd.DataFrame:
        """
        Loads the metadata set for subset specified in config.
        :return:
        """
        if not os.path.exists(self.data_paths.subsetFilePath):
            metadata = self.generate_metadata_csv()
        else:
            metadata = pd.read_csv(self.data_paths.subsetFilePath)

        return metadata

    @staticmethod
    def plot_genres_distribution(metadata: pd.DataFrame, title: str) -> None:
        """
        Plot distribution of genres in the metadata set.
        :param metadata: metadata set
        :param title: title of the plot
        :return:
        """

        print(metadata["genre_top"].value_counts())
        metadata["genre_top"].value_counts().plot(
            kind="pie", autopct="%1.1f%%", shadow=True
        )
        plt.title(title)
        plt.show()

    @staticmethod
    def plot_sizes(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> None:
        """
        Plot each dataset count on bar diagram.
        :param train: train set
        :param val: validation set
        :param test: test set
        :return:
        """

        plt.figure(figsize=(10, 5))
        plt.bar(["Train", "Validation", "Test"], [len(train), len(val), len(test)])
        plt.title("Dataset count")
        plt.show()

    @staticmethod
    def split_metadata(
        metadata: pd.DataFrame,
        train_ratio: float = 0.8,
        val_ratio: float = 0,
        test_ratio: float = 0,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Splits the metadata set into train, validation and test sets.
        :param metadata: DataFrame with the metadata set.
        :param train_ratio: Ratio of the train set.
        :param val_ratio: Ratio of the validation set.
        :param test_ratio: Ratio of the test set.
        :return: train, validation and test metadata sets.
        """

        if val_ratio == 0 and test_ratio == 0:
            val_ratio = test_ratio = (1 - train_ratio) / 2
        elif test_ratio == 0:
            test_ratio = 1 - train_ratio - val_ratio
        elif val_ratio == 0:
            val_ratio = 1 - train_ratio - test_ratio

        train_end = int(train_ratio * len(metadata))
        val_end = train_end + 1 + int(val_ratio * len(metadata))

        train = metadata.iloc[:train_end]
        val = metadata.iloc[train_end:val_end]
        test = metadata.iloc[val_end:]

        return train, val, test

    @staticmethod
    def split_metadata_uniform(
        metadata: pd.DataFrame,
        train_ratio: float = 0.8,
        val_ratio: float = 0,
        test_ratio: float = 0,
        add_val_to_train: bool = False,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Splits the metadata set into train, validation and test sets uniformly
        :param add_val_to_train: if true, the validation set is added to the train set
        :param metadata: dataFrame with the metadata set
        :param train_ratio: ratio of the train set
        :param val_ratio: ratio of the validation set
        :param test_ratio: ratio of the test set
        :param add_val_to_train: if true, validation set is added to test set
        :return: train, validation and test metadata sets
        """
        metadata_grouped = dict(tuple(metadata.groupby("genre_top")))

        train_list, val_list, test_list = [], [], []

        for genre in metadata_grouped:
            metadata_grouped[genre] = metadata_grouped[genre].sample(
                frac=1, random_state=42
            )
            train_genre, val_genre, test_genre = MetadataProcessor.split_metadata(
                metadata_grouped[genre],
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                test_ratio=test_ratio,
            )

            train_list.extend(train_genre.values.tolist())
            val_list.extend(val_genre.values.tolist())
            test_list.extend(test_genre.values.tolist())

        if add_val_to_train:
            train_list.extend(val_list)

        train = pd.DataFrame(train_list, columns=metadata.columns)
        val = pd.DataFrame(val_list, columns=metadata.columns)
        test = pd.DataFrame(test_list, columns=metadata.columns)

        return train, val, test
