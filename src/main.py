import os.path

from src.data_process.config_paths import DataPathsManager
from src.data_process.metadata_processor import MetadataProcessor
from src.data_process import spectrogram_generator
from src.data_process import spectrogram_augmenter
from src.data_process import spectrogram_debug
from src.data_process import dataset_analysis
from dataset_to_wav import convert_mp3_dataset_to_wav
from training.training_config import TrainingConfig
from training.training_manager import run_training

if __name__ == "__main__":
    data_paths = DataPathsManager()
    # Get metadata
    processor = MetadataProcessor()
    metadata = processor.get_metadata()

    if True:
        for SPLIT in range(1, 4):
            if SPLIT == 1:
                train, val, test = processor.split_metadata(metadata, 0.8, 0.1, 0.1)
                normalize = False
                augment = False
            elif SPLIT == 2:
                train, val, test = processor.split_metadata_uniform(
                    metadata, 0.8, 0.1, 0.1
                )
                normalize = True
                augment = True
            elif SPLIT == 3:
                train, val, test = processor.split_metadata_uniform(
                    metadata, 0.8, 0.1, 0.1, add_val_to_train=True
                )
                normalize = True
                augment = True

            # Plot metadata distribution
            processor.plot_metadata_distribution(metadata, "Metadata distribution")

            processor.plot_metadata_distribution(train, "TRAIN set distribution")
            processor.plot_metadata_distribution(val, "VAL set distribution")
            processor.plot_metadata_distribution(test, "TEST set distribution")
            processor.plot_train_val_test_counts(train, val, test)

            # Prepare dataset and generate spectrogram
            for dataset_metadata, path in (
                (train, data_paths.get_train_dataset_path(SPLIT)),
                (val, data_paths.get_val_dataset_path(SPLIT)),
                (test, data_paths.get_test_dataset_path(SPLIT)),
            ):
                if not os.path.exists(path):
                    spectrogram_generator.generate_all_spectrograms(
                        dataset_path=data_paths.datasetPath,
                        metadata=dataset_metadata,
                        save_path=path,
                        normalize=normalize,
                    )
            training_config = TrainingConfig(f"SPLIT nr.{SPLIT}")
            run_training(training_config, metadata, processor, data_paths, SPLIT)
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
