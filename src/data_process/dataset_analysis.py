import os
from typing import Dict

import numpy as np
import pandas as pd
from alive_progress import alive_bar
from matplotlib import pyplot as plt
from scipy.io.wavfile import write

from src.data_process import spectrogram_generator
from src.data_process.config_paths import DataPathsManager
from src.data_process.metadata_processor import MetadataProcessor


def get_index_of_silence(signal: np.ndarray, threshold: float = 0.1) -> float:
    """
    Return index of silence, which is proportion of silence in signal to the length of signal
    :param signal: signal
    :param threshold: threshold of silence
    :return:
    """

    # Normalize signal
    signal = signal / np.max(np.abs(signal))

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


def generate_silence_wav(path: str, duration_s: int, sample_rate: int = 44100) -> None:
    """
    Generate silence wav
    :param path: path to save wav
    :param duration_s: duration of wav in seconds
    :param sample_rate: sample rate of wav
    :return:
    """

    number_of_samples = duration_s * sample_rate

    data = np.zeros(number_of_samples)
    scaled = np.int16(data / np.max(np.abs(data)) * 32767)
    write(path, sample_rate, scaled)


def validate_ios(metadata: pd.DataFrame) -> None:
    """
    Validate index of silence on whole dataset.
    """
    data_paths = DataPathsManager()

    noise_name = "noise.wav"
    silence_name = "silence.wav"

    if not os.path.exists(noise_name):
        generate_noise_wav(noise_name, 10)

    if not os.path.exists(silence_name):
        generate_silence_wav(silence_name, 10)

    # Calculate index of silence for noise
    signal = spectrogram_generator.get_signal(noise_name)
    print_index_of_silence(noise_name, signal)

    # Calculate index of silence for silence
    signal = spectrogram_generator.get_signal(silence_name)
    print_index_of_silence(silence_name, signal)

    # Clear file ios_list.txt
    open("ios_list.txt", "w").close()

    ios_list = []
    tracks_number = 8000

    with alive_bar(tracks_number, title="Index of silence") as bar:
        for i in range(tracks_number):
            signal = spectrogram_generator.get_signal_by_index(
                data_paths.datasetPath, metadata, i
            )

            # Validate is signal was loaded
            if signal is None:
                bar()
                continue

            ios_list.append((i, get_index_of_silence(signal)))
            with open("ios_list.txt", "a") as file:
                file.write(f"{i};{get_index_of_silence(signal)}\n")
            bar()

    # Get n tracks with highest ios with index_of_silence
    n = int(tracks_number / 10)
    ios_list.sort(key=lambda x: x[1], reverse=True)
    print(f"{n} tracks with highest ios with index_of_silence:")
    for i in range(n):
        path = spectrogram_generator.index_to_file_path(
            data_paths.datasetPath, metadata, ios_list[i][0]
        )
        print(f"Track {ios_list[i][0]};{ios_list[i][1]}; {path}")


def get_genres_distances(metadata: pd.DataFrame, data_path: str) -> Dict[str, Dict]:
    """
    Calculate the distance between genres in the dataset
    :param metadata: metadata of the dataset
    :param data_path: path to the dataset
    :return: dictionary with the distance between genres
    """

    # Sort the data by genres, where metadata is list of tracks and their genres
    metadata_sorted = metadata.sort_values(by=["genre_top"])

    # Get spectrogram shape
    spectrogram_shape = spectrogram_generator.generate_spectrogram_by_index(
        data_path, metadata_sorted, 0
    ).shape
    genre_avg_spectrogram = {}

    # For each genre, calculate average spectrogram
    for genre_name in metadata_sorted["genre_top"].unique():
        genre_metadata = metadata_sorted[metadata_sorted["genre_top"] == genre_name]

        # For each track in genre, calculate spectrogram and divide by number of tracks in genre
        # Then add to genre_avg
        genre_avg = np.zeros(spectrogram_shape)
        error_count = 0
        for track in genre_metadata.index:
            try:
                spectrogram = spectrogram_generator.generate_spectrogram_by_index(
                    data_path, metadata_sorted, track
                )
                # Reshape spectrogram to size of genre_avg
                # Check if shape is lower than genre_avg, if so, append zeros
                if spectrogram.shape < genre_avg.shape:
                    spectrogram = np.append(
                        spectrogram,
                        np.zeros(genre_avg.shape - spectrogram.shape),
                        axis=0,
                    )

                # Check if shape is higher than genre_avg, if so, cut
                spectrogram = spectrogram[
                    : spectrogram_shape[0], : spectrogram_shape[1]
                ]
                genre_avg += spectrogram / len(genre_metadata)
            except RuntimeError:
                print("Error in track: " + str(metadata_sorted.iloc[track]["track_id"]))
                error_count += 1

        # Multiply by error_count to get the correct average
        genre_avg *= len(genre_metadata) / (len(genre_metadata) - error_count)
        genre_avg_spectrogram[genre_name] = genre_avg
        print("Finished class: " + genre_name)
        print(genre_avg)

    # Calculate distance between genres
    genres_distances: Dict[str, Dict] = {}
    for genre_name in genre_avg_spectrogram:
        genres_distances[genre_name] = {}
        for genre_avg_other in genre_avg_spectrogram:
            if genre_name != genre_avg_other:
                genres_distances[genre_name][genre_avg_other] = np.linalg.norm(
                    genre_avg_spectrogram[genre_name]
                    - genre_avg_spectrogram[genre_avg_other]
                )
                print("Finished class: " + genre_name + " to " + genre_avg_other)
                print(genres_distances[genre_name][genre_avg_other])

    return genres_distances


def save_genre_distances(metadata: pd.DataFrame, data_path: str) -> None:
    """
    Save the distances between genres to file
    :param metadata: metadata of the dataset
    :param data_path: path to the dataset
    """
    genres_distances = get_genres_distances(metadata, data_path)
    np.save("genres_distances.npy", genres_distances)


def plot_avg_tracks_per_artist_by_genre() -> None:
    """
    Plot the average number of tracks per artist per genre.
    """
    processor = MetadataProcessor()
    metadata = processor.load_full_metadata()

    # Get dataset specific metadata for tracks
    metadata = metadata[metadata["set", "subset"] <= "small"]

    # Change metadata scope to genre information only
    track_genre_metadata = metadata["track"]["genre_top"]
    track_artist_metadata = metadata["artist"]["name"]

    track_artist_genre_metadata = pd.concat(
        [track_genre_metadata, track_artist_metadata], axis=1
    )
    track_artist_genre_metadata_grouped = dict(
        tuple(track_artist_genre_metadata.groupby("genre_top"))
    )

    result_artists_per_genre = []
    result_tracks_per_artist_by_genre = []
    for genre in track_artist_genre_metadata_grouped:
        genre_group = track_artist_genre_metadata_grouped[genre]
        if len(genre_group) > 1:
            tracks_number = len(genre_group)
            artists_number = len(genre_group.value_counts())
            result_artists_per_genre.append((genre, artists_number))
            result_tracks_per_artist_by_genre.append(
                (genre, tracks_number / artists_number)
            )
            print(
                genre + ": " + str(len(genre_group) / len(genre_group.value_counts()))
            )

    result_df = pd.DataFrame(
        result_tracks_per_artist_by_genre, columns=["genre", "tracks_per_artist"]
    )
    print(result_df)
    result_df.plot(
        kind="bar",
        x="genre",
        y="tracks_per_artist",
        legend=False,
        title="Avg. tracks per artist by genre",
    )
    plt.xlabel("Genre")
    plt.ylabel("Avg. tracks per Artist")
    plt.tight_layout()
    plt.show()

    result_df = pd.DataFrame(
        result_artists_per_genre, columns=["genre", "number_of_artists"]
    )
    print(result_df)
    result_df.plot(
        kind="bar",
        x="genre",
        y="number_of_artists",
        legend=False,
        title="Number of artists by genre",
    )
    plt.xlabel("Genre")
    plt.ylabel("Number of artists by genre")
    plt.tight_layout()
    plt.show()
