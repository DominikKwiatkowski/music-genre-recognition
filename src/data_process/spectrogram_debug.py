import librosa
import librosa.display
import numpy as np
from matplotlib import pyplot as plt

from src.data_process import spectrogram_augmenter
from src.data_process.spectrogram_generator import default_sample_rate


def plot_spectrogram(
        spectrogram: np.ndarray, sample_rate: int = default_sample_rate
) -> None:
    """
    Plot spectrogram.
    :param spectrogram: Spectrogram
    :param sample_rate: Sample rate
    :return: None
    """

    plt.figure(figsize=(14, 5))
    librosa.display.specshow(spectrogram, sr=sample_rate, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+1.0f dB')
    plt.show()


def validate_noise_overlay(spectrogram: np.ndarray) -> None:
    """
    Test noise overlay.
    :return: None
    """

    # Plot spectrogram
    plot_spectrogram(spectrogram)

    # Add noise and plot
    spectrogram_noised = spectrogram_augmenter.noise_overlay(spectrogram, 1, 0.1)
    plot_spectrogram(spectrogram_noised)
