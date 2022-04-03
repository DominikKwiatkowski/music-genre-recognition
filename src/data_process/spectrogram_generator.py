import os
import librosa
import librosa.display
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt

default_sample_rate = 44100


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

    # Convert track_id to filename, which is id aligned to 6 "0"s + ".mp3"
    file_name = str(track_id).zfill(6) + ".mp3"

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

    if not file_path.endswith(".mp3"):
        print("File is not a mp3 file")
        return None

    # Load file
    signal, sample_rate = librosa.load(file_path, sr=default_sample_rate)

    # Duplicate mono signal to stereo
    if signal.shape[0] == 1:
        torch.cat([signal, signal])

    # Generate spectrogram data
    sgram = librosa.stft(signal)
    sgram_mag, _ = librosa.magphase(sgram)
    mel_scale_sgram = librosa.feature.melspectrogram(S=sgram_mag, sr=sample_rate)
    mel_sgram = librosa.amplitude_to_db(mel_scale_sgram, ref=np.min)

    return mel_sgram


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
