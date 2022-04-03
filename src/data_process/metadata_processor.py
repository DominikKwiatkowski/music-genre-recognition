import configparser
import os
import pandas as pd
import ast
import matplotlib.pyplot as plt

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
