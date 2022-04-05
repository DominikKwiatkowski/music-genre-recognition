import numpy as np
from typing import Dict
from src.data_process.spectrogram_generator import generate_spectrogram_by_index
import pandas as pd
import matplotlib.pyplot as plt
from src.data_process.metadata_processor import MetadataProcessor

class AnalyzeData:
    def __init__(self, data_path, metadata):
        self.data_path = data_path
        self.metadata = metadata

    def find_class_distances(self) -> None:
        """
        Find the distance between each class
        """
        # Sort the data by class, metadata is array of track and genre_top
        self.metadata = self.metadata.sort_values(by=["genre_top"])
        # Get spectogram sizes
        spectogram_shape = generate_spectrogram_by_index(
            self.data_path, self.metadata, 0
        ).shape
        genre_avg_spectogram = dict()
        # for each class, calculate avarage spectogram
        for class_name in self.metadata["genre_top"].unique():
            class_data = self.metadata[self.metadata["genre_top"] == class_name]
            # for each track in class, calculate spectogram and divide by class size, then add to class_avarage
            class_avarage = np.zeros(spectogram_shape)
            error_count = 0
            for track in class_data.index:
                try:
                    spectogram = generate_spectrogram_by_index(
                        self.data_path, self.metadata, track
                    )
                    # Reshape spectogram to size of class_avarage
                    # check if shape is lower than class_avarage, if so, append zeros
                    if spectogram.shape < class_avarage.shape:
                        spectogram = np.append(
                            spectogram,
                            np.zeros(class_avarage.shape - spectogram.shape),
                            axis=0,
                        )
                    # check if shape is higher than class_avarage, if so, cut
                    spectogram = spectogram[
                                 : spectogram_shape[0], : spectogram_shape[1]
                                 ]
                    class_avarage += spectogram / len(class_data)
                except RuntimeError:
                    print(
                        "Error in track: " + str(self.metadata.iloc[track]["track_id"])
                    )
                    error_count += 1
            # multiple by error_count to get the correct avarage
            class_avarage *= len(class_data) / (len(class_data) - error_count)
            genre_avg_spectogram[class_name] = class_avarage
            print("Finished class: " + class_name)
            print(class_avarage)

        # calculate distance between classes
        class_distances: Dict[str, Dict] = dict()
        for class_name in genre_avg_spectogram:
            class_distances[class_name] = dict()
            for class_name2 in genre_avg_spectogram:
                if class_name != class_name2:
                    class_distances[class_name][class_name2] = np.linalg.norm(
                        genre_avg_spectogram[class_name]
                        - genre_avg_spectogram[class_name2]
                    )
                    print("Finished class: " + class_name + " to " + class_name2)
                    print(class_distances[class_name][class_name2])
        # save class_distances to file
        np.save("class_distances.npy", class_distances)

    @staticmethod
    def plot_avg_tracks_per_artist_by_genre() -> None:
        processor = MetadataProcessor()
        metadata = processor.load_full_metadata()

        # Get dataset specific metadata for tracks
        metadata = metadata[
            metadata["set", "subset"] <= "small"
            ]

        # Change metadata scope to genre information only
        track_genre_metadata = metadata['track']['genre_top']
        track_artist_metadata = metadata['artist']['name']

        track_artist_genre_metadata = pd.concat([track_genre_metadata, track_artist_metadata], axis=1)
        track_artist_genre_metadata_grouped = dict(tuple(track_artist_genre_metadata.groupby('genre_top')))

        result_artists_per_genre = []
        result_tracks_per_artist_by_genre = []
        for genre in track_artist_genre_metadata_grouped:
            genre_group = track_artist_genre_metadata_grouped[genre]
            if len(genre_group) > 1:
                tracks_number = len(genre_group)
                artists_number = len(genre_group.value_counts())
                result_artists_per_genre.append((genre, artists_number))
                result_tracks_per_artist_by_genre.append((genre, tracks_number / artists_number))
                print(genre + ": " + str(len(genre_group) / len(genre_group.value_counts())))

        result_df = pd.DataFrame(result_tracks_per_artist_by_genre, columns=['genre', 'tracks_per_artist'])
        print(result_df)
        result_df.plot(kind='bar', x='genre', y='tracks_per_artist', legend=False,
                       title='Avg. tracks per artist by genre')
        plt.xlabel('Genre')
        plt.ylabel('Avg. tracks per Artist')
        plt.tight_layout()
        plt.show()

        result_df = pd.DataFrame(result_artists_per_genre, columns=['genre', 'number_of_artists'])
        print(result_df)
        result_df.plot(kind='bar', x='genre', y='number_of_artists', legend=False,
                       title='Number of artists by genre')
        plt.xlabel('Genre')
        plt.ylabel('Number of artists by genre')
        plt.tight_layout()
        plt.show()
