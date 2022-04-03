import configparser
import os
import pandas as pd
import ast
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from src.data_process.config_paths import DataPathsManager


class MetadataProcessor:
    def __init__(self):
        self.data_paths = DataPathsManager()

    def load_full_metadata(self) -> pd.DataFrame:
        """
        Loads the label file. Source: https://github.com/mdeff/fma/blob/master/utils.py
        :return: DataFrame with the trackMetadata
        """
        trackMetadata = pd.read_csv(self.data_paths.labelFilePath, index_col=0, header=[0, 1])
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
        subsetMetadata = trackMetadata[trackMetadata["set", "subset"] <= self.data_paths.datasetName]

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
    def plot_metadata_distribution(metadata: pd.DataFrame) -> None:
        """
        Plots the data set.
        :return:
        """
        # show genre count plot
        print(metadata["genre_top"].value_counts())

        metadata["genre_top"].value_counts().plot(
            kind="pie", autopct="%1.1f%%", shadow=True
        )

        plt.show()

    @staticmethod
    def split_metadata(metadata, train_ratio=0.8, val_ratio=0, test_ratio=0) \
            -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
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

        if test_ratio == 0:
            test_ratio = 1 - train_ratio - val_ratio

        train_metadata, remainder_metadata = train_test_split(metadata, train_size=train_ratio, shuffle=True)
        val_metadata, test_metadata = train_test_split(remainder_metadata,
                                                       train_size=(val_ratio / (val_ratio + test_ratio)), shuffle=True)
        return train_metadata, val_metadata, test_metadata

    @staticmethod
    def split_metadata_uniform(metadata, train_ratio=0.8, val_ratio=0, test_ratio=0, add_val_to_test=False) \
            -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
        """
        Splits the metadata set into train, validation and test sets uniformly.
        :param metadata: DataFrame with the metadata set.
        :param train_ratio: Ratio of the train set.
        :param val_ratio: Ratio of the validation set.
        :param test_ratio: Ratio of the test set.
        :return: train, validation and test metadata sets.
        """
        metadata_grouped = dict(tuple(metadata.groupby('genre_top')))

        train_metadata, val_metadata, test_metadata = [], [], []

        for genre in metadata_grouped:
            train_metadata_genre, val_metadata_genre, test_metadata_genre = MetadataProcessor.split_metadata(
                metadata_grouped[genre],
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                test_ratio=test_ratio)
            train_metadata.extend(train_metadata_genre.values.tolist())
            if add_val_to_test:
                test_metadata.extend(val_metadata_genre.values.tolist())
            val_metadata.extend(val_metadata_genre.values.tolist())
            test_metadata.extend(test_metadata_genre.values.tolist())

        return pd.DataFrame(train_metadata, columns=metadata.columns), \
               pd.DataFrame(val_metadata, columns=metadata.columns), \
               pd.DataFrame(test_metadata, columns=metadata.columns)
