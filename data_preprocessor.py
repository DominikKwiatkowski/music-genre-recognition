import configparser
import numpy as np
import os
import pandas as pd
import ast
import matplotlib.pyplot as plt
import librosa
import librosa.display
import torch
import random
from torchaudio import transforms


class DataPreprocessor:
    def __init__(self):
        # Load config file
        self.data = None
        self.config = configparser.ConfigParser()
        self.config.read("config.ini")

        # Generate paths for data set
        self.subsetFilePath = self.config.get(
            "METADATA", "METADATA_PATH"
        ) + self.config.get("METADATA", "SUBSET_FILE")
        self.labelFilePath = self.config.get(
            "METADATA", "METADATA_PATH"
        ) + self.config.get("METADATA", "LABEL_FILE")

        self.datasetName = self.config.get("DATASET", "DATASET_NAME")
        self.datasetPath = (
            self.config.get("DATASET", "DATASET_PATH")
            + self.config.get("DATASET", "DATASET_PREFIX")
            + self.config.get("DATASET", "DATASET_NAME")
        )

    def createDatasetCsv(self) -> pd.DataFrame:
        """
        Creates a csv file with the data set. Only track id and genre are saved.
        :return:
        """
        # Read label file, which is csv file
        trackMetadata = self.loadTracksMetadata()

        # Get dataset specific metadata for tracks
        subsetMetadata = trackMetadata[
            trackMetadata["set", "subset"] <= self.datasetName
        ]

        # Change metadata scope to genre information only
        subsetMetadata = subsetMetadata["track"]["genre_top"]

        # Save genres to csv
        subsetMetadata.to_csv(self.subsetFilePath)

        return subsetMetadata

    def loadTracksMetadata(self) -> pd.DataFrame:
        """
        Loads the label file. Source: https://github.com/mdeff/fma/blob/master/utils.py
        :return: DataFrame with the trackMetadata
        """
        trackMetadata = pd.read_csv(self.labelFilePath, index_col=0, header=[0, 1])
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

    def loadData(self) -> None:
        """
        Loads the data set for selected subset.
        :return:
        """
        if not os.path.exists(self.subsetFilePath):
            self.data = self.createDatasetCsv()
        else:
            self.data = pd.read_csv(self.subsetFilePath)

    def plotData(self) -> None:
        """
        Plots the data set.
        :return:
        """
        # show genre count plot
        print(self.data["genre_top"].value_counts())

        self.data["genre_top"].value_counts().plot(
            kind="pie", autopct="%1.1f%%", shadow=True
        )

        plt.show()
