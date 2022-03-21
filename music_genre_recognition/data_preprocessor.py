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

        # save plot
        plt.show()

    def create_spectrograms(self) -> None:
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
                    self.__create_spectrogram(os.path.join(root, file))

    def __create_spectrogram(self, file_path: str) -> None:
        """
        Creates a spectogram for the given file.
        :param file_path: Path to the file
        :return:
        """
        # Load file
        signal, sample_rate = librosa.load(file_path, sr=44100)
        if signal.shape[0] == 1:
            # duplicate first channel
            torch.cat([signal, signal])

        # create spectogram
        plt.figure(figsize=(14, 5))
        sgram = librosa.stft(signal)
        sgram_mag, _ = librosa.magphase(sgram)
        mel_scale_sgram = librosa.feature.melspectrogram(S=sgram_mag, sr=sample_rate)
        mel_sgram = librosa.amplitude_to_db(mel_scale_sgram, ref=np.min)
        librosa.display.specshow(mel_sgram, sr=sample_rate, x_axis="time", y_axis="mel")
        plt.show()

        # create augmented spectogram
        spectrogram = self.spectro_augment(mel_sgram)
        librosa.display.specshow(
            spectrogram, sr=sample_rate, x_axis="time", y_axis="mel"
        )
        plt.show()

    @staticmethod
    def change_length(signal: np.ndarray, sample_rate: int, max_ms: int) -> np.ndarray:
        sig_len = signal.size
        max_len = sample_rate // 1000 * max_ms

        if sig_len > max_len:
            # Truncate the signal to the given length
            signal = signal[:max_len]

        elif sig_len < max_len:
            # Length of padding to add at the beginning and end of the signal
            pad_begin_len = random.randint(0, max_len - sig_len)
            pad_end_len = max_len - sig_len - pad_begin_len

            # Pad with 0s
            numpy_begin = np.zeros(pad_begin_len)
            numpy_end = np.zeros(pad_end_len)
            signal = np.concatenate((numpy_begin, signal, numpy_end))

        return signal

    @staticmethod
    def time_shift(signal: np.ndarray, shift_limit: float = 0.1) -> np.ndarray:
        sig_len = signal.size
        shift_amt = int(random.random() * shift_limit * sig_len)
        return np.roll(signal, shift_amt)

    @staticmethod
    def spectro_augment(
        signal: np.ndarray,
        max_mask_pct: float = 0.1,
        n_freq_masks: int = 1,
        n_time_masks: int = 1,
    ) -> np.ndarray:
        n_mels, n_steps = signal.shape
        mask_value = signal.mean()
        aug_signal = signal

        freq_mask_param = max_mask_pct * n_mels
        for _ in range(n_freq_masks):
            f_idx = np.random.randint(0, n_mels)
            f_width = np.random.randint(1, int(freq_mask_param))
            aug_signal[f_idx : f_idx + f_width, :] = mask_value

        time_mask_param = max_mask_pct * n_steps
        for _ in range(n_time_masks):
            t_idx = np.random.randint(0, n_steps)
            t_width = np.random.randint(1, int(time_mask_param))
            aug_signal[:, t_idx : t_idx + t_width] = mask_value

        return aug_signal
