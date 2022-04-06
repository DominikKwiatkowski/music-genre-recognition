import os
import librosa
import librosa.display
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt

default_sample_rate = 44100

def plot_spectrogram(
    spectrogram: np.ndarray, sample_rate: int = default_sample_rate
) -> None:
    """
    Plots spectrogram.
    :param spectrogram: Spectrogram
    :param sample_rate: Sample rate
    :return: None
    """

    plt.figure(figsize=(14, 5))
    librosa.display.specshow(spectrogram, sr=sample_rate, x_axis="time", y_axis="mel")
    plt.colorbar(format='%+2.0f dB')
    plt.show()


def index_to_file_path(dataset_path: str, metadata: pd.DataFrame, index: int) -> str:
    """
    Creates spectrogram for file in the dataset.
    :return: Spectrogram
    """

    # Check if index is valid
    if index < 0 or index >= len(metadata):
        print("Index out of range")
        return ""

    # Get row of index
    track_id = metadata.iloc[index]["track_id"]

    # Convert track_id to folder name
    folder_name = str(int(track_id / 1000)).zfill(3)

    # Convert track_id to filename, which is id aligned to 6 "0"s + ".wav"
    file_name = str(track_id).zfill(6) + ".wav"

    return os.path.join(dataset_path, folder_name, file_name)


def generate_spectrogram(file_path: str) -> np.ndarray:
    """
    Creates a spectrogram for the given file.
    param file_path: Path to the file
    :return: Spectrogram
    """

    # Check if file exists and is a "mp3" file
    if not os.path.isfile(file_path):
        print("File does not exist")
        return None

    if not file_path.endswith(".wav"):
        print("File is not a mp3 file")
        return None

    # Load file
    signal, sample_rate = librosa.load(file_path, sr=default_sample_rate)

    # Duplicate mono signal to stereo
    if signal.shape[0] == 1:
        torch.cat([signal, signal])

    # Generate spectrogram data
    sgram = librosa.stft(signal)
    sgram_mag = np.abs(sgram)
    mel_scale_sgram = librosa.feature.melspectrogram(S=sgram_mag, sr=sample_rate, power=1)
    mel_sgram = librosa.amplitude_to_db(mel_scale_sgram, ref=np.max)

    return mel_sgram


def normalize_spectrogram(spectrogram: np.ndarray) -> np.ndarray:
    """
    Normalizes mel dB spectrogram to [-1; 1] range.
    :return: Normalized spectrogram
    """
    norm_spectrogram = 2 * librosa.util.normalize(spectrogram) - 1
    return norm_spectrogram


def generate_random_spectrogram(
        dataset_path: str, metadata: pd.DataFrame
) -> np.ndarray:
    """
    Creates spectrogram for random file in the dataset.
    :return: Spectrogram
    """

    # Randomize track index
    random_index = np.random.randint(0, len(metadata))
    file_path = index_to_file_path(dataset_path, metadata, random_index)

    return generate_spectrogram(file_path)


def generate_spectrogram_by_index(
        dataset_path: str, metadata: pd.DataFrame, index: int
) -> np.ndarray:
    """
    Creates spectrogram for file in the dataset.
    :return: Spectrogram
    """

    # Generate file path
    file_path = index_to_file_path(dataset_path, metadata, index)

    return generate_spectrogram(file_path)
