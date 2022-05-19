import os

import librosa
import librosa.display
import numpy as np
import pandas as pd
import torch
from alive_progress import alive_bar

import src.data_process.spectrogram_debug as sdbg

default_sample_rate = 44100

track_ext = ".wav"


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
    file_name = str(track_id).zfill(6) + track_ext

    return os.path.join(dataset_path, folder_name, file_name)


def get_signal(file_path: str) -> np.ndarray:
    """
    Loads signal from file.
    :return: Signal
    """

    # Check if file exists
    if not os.path.isfile(file_path):
        print("File does not exist")
        return None

    # Load file
    try:
        signal, sample_rate = librosa.load(file_path, sr=None)
    except RuntimeError:
        print("Error loading file")
        return None

    return signal


def get_signal_by_index(
    dataset_path: str, metadata: pd.DataFrame, index: int
) -> np.ndarray:
    """
    Loads signal from file.
    :return: Signal
    """

    # Generate file path
    file_path = index_to_file_path(dataset_path, metadata, index)

    return get_signal(file_path)


def generate_spectrogram(file_path: str, spectro_height: int = 128) -> np.ndarray:
    """
    Creates a spectrogram for the given file.
    param file_path: Path to the file
    :return: Spectrogram
    """

    # Check if file exists and has correct extension
    if not os.path.isfile(file_path):
        print("File does not exist")
        return None

    if not file_path.endswith(track_ext):
        print(f"File is not a {track_ext} file")
        return None

    # Load file
    signal, sample_rate = librosa.load(file_path, sr=default_sample_rate)

    # Duplicate mono signal to stereo
    if signal.shape[0] == 1:
        torch.cat([signal, signal])

    # Generate spectrogram data
    sgram = librosa.stft(signal)
    sgram_mag = np.abs(sgram)
    mel_scale_sgram = librosa.feature.melspectrogram(
        S=sgram_mag, sr=sample_rate, power=1, n_mels=spectro_height
    )
    mel_sgram = librosa.amplitude_to_db(mel_scale_sgram, ref=np.min)

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
    :param dataset_path: path to the dataset
    :param metadata: metadata of the dataset
    :param index: index of the track
    :return: Spectrogram
    """

    # Generate file path from index
    file_path = index_to_file_path(dataset_path, metadata, index)

    return generate_spectrogram(file_path)


def generate_all_spectrograms(
    dataset_path: str,
    metadata: pd.DataFrame,
    save_path: str,
    normalize: bool,
    spectro_height: int = 128,
) -> None:
    """
    Create spectrograms for all files in the dataset and save them to disk
    :param dataset_path: path to the dataset
    :param metadata: metadata of the dataset
    :param save_path: path to save spectrograms
    :param normalize: whether to normalize spectrograms
    :param spectro_height: height of the spectrogram
    """

    # Create save path
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Generate and save spectrogram
    with alive_bar(
        len(metadata), title=f"Preparing spectrograms for {save_path}"
    ) as bar:
        for index in range(len(metadata)):
            try:
                file_path = index_to_file_path(dataset_path, metadata, index)
                spectrogram = generate_spectrogram(file_path, spectro_height)
                if normalize:
                    spectrogram = normalize_spectrogram(spectrogram)

                filename = str(metadata.iloc[index]["track_id"])
                save_file_path = os.path.join(save_path, filename)
                np.save(save_file_path, spectrogram)
            except TypeError:
                print(f"Error generating spectrogram for file {index}")

            bar()
