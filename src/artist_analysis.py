from src.data_process.config_paths import DataPathsManager
from src.data_process.metadata_processor import MetadataProcessor
import src.data_process.spectrogram_generator as spectrogram_generator
import src.data_process.spectrogram_augmenter as spectrogram_augmenter
import src.data_process.spectrogram_debug as spectrogram_debug
import src.data_process.analyze_data as ad
if __name__ == "__main__":
    ad.AnalyzeData.plot_avg_tracks_per_artist_by_genre()

