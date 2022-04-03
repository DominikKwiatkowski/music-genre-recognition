import random

import numpy as np


def time_shift(signal: np.ndarray, shift_limit: float = 0.1) -> np.ndarray:
    """
    Shift signal by a random amount.
    signal: ndarray of shape (n_samples, n_channels)
    shift_limit: maximum percentage of shift
    : return: shifted signal
    """
    sig_len = signal.size
    shift_amt = int(random.random() * shift_limit * sig_len)
    return np.roll(signal, shift_amt)


def augment_spectrogram(
        spectrogram: np.ndarray,
        max_mask_pct: float = 0.1,
        n_freq_masks: int = 1,
        n_time_masks: int = 1,
) -> np.ndarray:
    """
    Augment spectrogram by masking out frequency and time bins.
    signal: ndarray of shape (n_samples, n_channels)
    max_mask_pct: maximum percentage of bins to mask out
    n_freq_masks: number of frequency masks to apply
    n_time_masks: number of time masks to apply
    : return: augmented signal
    """

    n_mels, n_steps = spectrogram.shape
    mask_value = spectrogram.mean()
    aug_signal = spectrogram

    freq_mask_param = max_mask_pct * n_mels
    for _ in range(n_freq_masks):
        f_idx = np.random.randint(0, n_mels)
        f_width = np.random.randint(1, int(freq_mask_param))
        aug_signal[f_idx: f_idx + f_width, :] = mask_value

    time_mask_param = max_mask_pct * n_steps
    for _ in range(n_time_masks):
        t_idx = np.random.randint(0, n_steps)
        t_width = np.random.randint(1, int(time_mask_param))
        aug_signal[:, t_idx: t_idx + t_width] = mask_value

    return aug_signal
