from src.data_process.config_paths import DataPathsManager
from src.data_process.metadata_processor import MetadataProcessor
import src.data_process.spectrogram_generator as spectrogram_generator
import src.data_process.spectrogram_augmenter as spectrogram_augmenter
import src.data_process.spectrogram_debug as spectrogram_debug

if __name__ == "__main__":
    data_paths = DataPathsManager()

    # Get metadata
    processor = MetadataProcessor()
    metadata = processor.get_metadata()

    # Plot metadata distribution
    processor.plot_metadata_distribution(metadata)

    # Generate random spectrogram
    spectrogram = spectrogram_generator.generate_spectrogram_by_index(data_paths.datasetPath, metadata, 10)
    spectrogram_debug.plot_spectrogram(spectrogram)

    augmented_spectrogram = spectrogram_augmenter.augment_spectrogram(spectrogram, 0.1, 5, 0)
    spectrogram_debug.plot_spectrogram(augmented_spectrogram)
