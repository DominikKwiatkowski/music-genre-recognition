import numpy as np
import pandas as pd
from alive_progress import alive_bar
from scipy.io.wavfile import write

from src.data_process import spectrogram_generator, spectrogram_debug
from src.data_process.config_paths import DataPathsManager


def get_index_of_silence(signal: np.ndarray, threshold: float = 0.1) -> float:
    """
        Return index of silence, which is proportion of silence in signal to the length of signal
        :param signal: signal
        :param threshold: threshold of silence
        :return:
    """

    # Get index of silence
    index_of_silence = np.where(np.abs(signal) < threshold)[0]

    # Return proportion of silence
    return len(index_of_silence) / len(signal)


def print_index_of_silence(name: str, signal: np.ndarray) -> None:
    """
        Print index of silence in file
        :param name: name of file
        :param signal: signal
        :return:
    """

    # Get index of silence
    index_of_silence = get_index_of_silence(signal)

    # Print index of silence
    print(f"Track: {name}, index_of_silence: {index_of_silence}")


def generate_noise_wav(path: str, duration_s: int, sample_rate: int = 44100) -> None:
    """
        Generate noise wav
        :param path: path to save wav
        :param duration_s: duration of wav in seconds
        :param sample_rate: sample rate of wav
        :return:
    """

    number_of_samples = duration_s * sample_rate

    data = np.random.uniform(-1, 1, number_of_samples)
    scaled = np.int16(data / np.max(np.abs(data)) * 32767)
    write(path, sample_rate, scaled)


def validate_ios(metadata: pd.DataFrame) -> None:
    """
    Validate index of silence on whole dataset.
    """
    data_paths = DataPathsManager()

    # generate_noise_wav("noise.wav", 30)

    # Calculate index of silence for silence
    signal = spectrogram_generator.get_signal("30-seconds-of-silence.mp3")
    print_index_of_silence("silence", signal)

    # Calculate index of silence for noise
    signal = spectrogram_generator.get_signal("noise.wav")
    print_index_of_silence("noise", signal)

    # Clear file ios_list.txt
    open("ios_list.txt", "w").close()

    ios_list = []
    tracks_number = 20

    with alive_bar(tracks_number, title='Index of silence') as bar:
        for i in range(tracks_number):
            signal = spectrogram_generator.get_signal_by_index(data_paths.datasetPath, metadata, i)
            if signal is None:
                continue

            ios_list.append((i, get_index_of_silence(signal)))
            with open("ios_list.txt", "a") as file:
                file.write(f"{i} {get_index_of_silence(signal)}\n")
            bar()

    # Get n tracks with highest ios with index_of_silence
    n = 100
    ios_list.sort(key=lambda x: x[1], reverse=True)
    print(f"{n} tracks with highest ios with index_of_silence:")
    for i in range(n):
        print(f"Track {ios_list[i][0]};{ios_list[i][1]}")


def validate_sampling(filepath: str) -> None:
    """
    Validate if sampling the spectrogram will be different from sampling the signal.
    """
    sample_rate: int = 44100

    spectrogram = spectrogram_generator.generate_spectrogram(filepath)

    # Plot whole spectrogram
    spectrogram_debug.plot_spectrogram(spectrogram, sample_rate)

    # Get spectrogram dimensions
    spectrogram_dimensions = spectrogram.shape

    # Plot two halves of spectrogram
    spectrogram_debug.plot_spectrogram(spectrogram[:, :spectrogram_dimensions[1] // 2], sample_rate)
    spectrogram_debug.plot_spectrogram(spectrogram[:, spectrogram_dimensions[1] // 2:], sample_rate)

    # Calculate half-spectrogram
    half_spectrogram_1 = spectrogram_generator.generate_half_spectrogram(filepath, 0)
    half_spectrogram_2 = spectrogram_generator.generate_half_spectrogram(filepath, 1)

    # Plot half-spectrogram
    spectrogram_debug.plot_spectrogram(half_spectrogram_1, sample_rate)
    spectrogram_debug.plot_spectrogram(half_spectrogram_2, sample_rate)

