from src.data_process.config_paths import DataPathsManager
from src.data_process.metadata_processor import MetadataProcessor
from src.data_process import spectrogram_generator
from src.data_process import spectrogram_augmenter
from src.data_process import spectrogram_debug
from src.data_process import dataset_analysis
from dataset_to_wav import convert_mp3_dataset_to_wav
from training.training_config import TrainingConfig
from training.training_manager import run_training

from prepare_data import prepare_data

if __name__ == "__main__":
    data_paths = DataPathsManager()
    processor = MetadataProcessor()
    metadata = processor.get_metadata()

    if False:
        # Generate random spectrogram
        spectrogram = spectrogram_generator.generate_spectrogram_by_index(
            data_paths.datasetPath, metadata, 10
        )
        spectrogram_debug.plot_spectrogram(spectrogram)

        augmented_spectrogram = spectrogram_augmenter.mask_spectrogram(
            spectrogram, 0.1, 5, 0
        )
        spectrogram_debug.plot_spectrogram(augmented_spectrogram)

    if False:
        # Validate shape of the spectrogram
        dataset_analysis.validate_shape()
