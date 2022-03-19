import configparser
import numpy as np
import os
import pandas as pd
import ast
import matplotlib.pyplot as plt
import librosa
import librosa.display


class DataPreprocessor:
    def __init__(self):
        # Load config file
        self.data = None
        self.config = configparser.ConfigParser()
        self.config.read("config.ini")

    def create_data_set_csv(self) -> None:
        """
        Creates a csv file with the data set. Only track id and genre are saved.
        :return:
        """
        # Read label file, which is csv file
        tracks = self.__load_tracks()
        tracks = tracks[
            tracks["set", "subset"] <= self.config.get("PATH", "SUBSET_NAME")
        ]
        tracks = tracks["track"]
        # Only keep the track id and genre
        tracks = tracks["genre_top"]

        tracks.to_csv(self.config.get("PATH", "SUBSET_FILE"))

    def __load_tracks(self) -> pd.DataFrame:
        """
        Loads the label file. Source: https://github.com/mdeff/fma/blob/master/utils.py
        :return: DataFrame with the tracks
        """
        filepath = self.config.get("PATH", "LABEL_PATH")
        tracks = pd.read_csv(filepath, index_col=0, header=[0, 1])
        COLUMNS = [
            ("track", "tags"),
            ("album", "tags"),
            ("artist", "tags"),
            ("track", "genres"),
            ("track", "genres_all"),
        ]
        for column in COLUMNS:
            tracks[column] = tracks[column].map(ast.literal_eval)

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
            tracks[column] = pd.to_datetime(tracks[column])

        SUBSETS = ("small", "medium", "large")
        try:
            tracks["set", "subset"] = tracks["set", "subset"].astype(
                "category", categories=SUBSETS, ordered=True
            )
        except (ValueError, TypeError):
            # the categories and ordered arguments were removed in pandas 0.25
            tracks["set", "subset"] = tracks["set", "subset"].astype(
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
            tracks[column] = tracks[column].astype("category")

        return tracks

    def load_data(self) -> None:
        """
        Loads the data set for selected subset.
        :return:
        """
        self.data = pd.read_csv(self.config.get("PATH", "SUBSET_FILE"))

    def plot_data(self) -> None:
        """
        Plots the data set.
        :return:
        """
        # show genre count plot
        print(self.data["genre_top"].value_counts())
        self.data["genre_top"].value_counts().plot(
            kind="pie", autopct="%1.1f%%", shadow=True
        )
        # Put x axis on top
        plt.xticks(rotation=90)
        # save plot
        plt.show()

    def create_spectograms(self) -> None:
        """
        Creates spectograms for each track in the data set.
        :return:
        """
        # Load data path from config.ini
        data_path = self.config.get("PATH", "DATA_PATH")

        # For each track in the data_path
        for root, dirs, files in os.walk(data_path):
            for file in files:
                # If the file is a wav file
                if file.endswith(".mp3"):
                    # Create spectogram
                    self.__create_spectogram(os.path.join(root, file))

    def __create_spectogram(self, file_path: str) -> None:
        """
        Creates a spectogram for the given file.
        :param file_path: Path to the file
        :return:
        """
        # Load file
        samples, sample_rate = librosa.load(file_path, sr=None)
        plt.figure(figsize=(14, 5))
        sgram = librosa.stft(samples)
        sgram_mag, _ = librosa.magphase(sgram)
        mel_scale_sgram = librosa.feature.melspectrogram(S=sgram_mag, sr=sample_rate)
        mel_sgram = librosa.amplitude_to_db(mel_scale_sgram, ref=np.min)
        librosa.display.specshow(mel_sgram, sr=sample_rate, x_axis="time", y_axis="mel")
        plt.savefig(file_path.replace(".mp3", ".png"))
