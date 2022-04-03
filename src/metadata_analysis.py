from src.data_process.config_paths import DataPathsManager
from src.data_process.metadata_processor import MetadataProcessor
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":
    data_paths = DataPathsManager()

    # Get full metadata
    processor = MetadataProcessor()

    metadata = processor.load_full_metadata()

    # Get dataset specific metadata for tracks
    subsetMetadata = metadata[
        metadata["set", "subset"] <= "small"
        ]

    # Change metadata scope to genre information only
    track_genre_metadata = subsetMetadata['track']['genre_top']
    track_artist_metadata = subsetMetadata['artist']['name']

    track_artist_genre_metadata = pd.concat([track_genre_metadata, track_artist_metadata], axis=1)
    track_artist_genre_metadata_grouped = dict(tuple(track_artist_genre_metadata.groupby('genre_top')))

    result = []
    for genre in track_artist_genre_metadata_grouped:
        genre_group = track_artist_genre_metadata_grouped[genre]
        if len(genre_group) > 1:
            result.append((genre, len(genre_group)/len(genre_group.value_counts())))
            print(genre + ": " + str(len(genre_group)/len(genre_group.value_counts())))

    result_df = pd.DataFrame(result, columns=['genre', 'tracks_per_artist'])
    print(result_df)
    result_df.plot(kind='bar', x='genre', y='tracks_per_artist', legend=False, title='Avg. tracks per artist by genre')
    plt.xlabel('Genre')
    plt.ylabel('Avg. tracks per Artist')
    plt.tight_layout()
    plt.show()

