from src.data_process.config_paths import DataPathsManager
from src.data_process.metadata_processor import MetadataProcessor
from src.data_process import spectrogram_generator
from src.data_process import spectrogram_augmenter
from src.data_process import spectrogram_debug
from src.data_process import dataset_analysis

if __name__ == "__main__":
    data_paths = DataPathsManager()

    # Get metadata
    processor = MetadataProcessor()
    metadata = processor.get_metadata()

    if False:
        SPLIT = 3

        if SPLIT == 1:
            train, val, test = processor.split_metadata(metadata, 0.8, 0.1, 0.1)
            normalize = False
            augment = False
        elif SPLIT == 2:
            train, val, test = processor.split_metadata_uniform(metadata, 0.8, 0.1, 0.1)
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
                (train, data_paths.trainDatasetPath),
                (val, data_paths.trainDatasetPath),
                (test, data_paths.testDatasetPath),
        ):
            spectrogram_generator.generate_all_spectrograms(
                dataset_path=data_paths.datasetPath,
                metadata=dataset_metadata,
                save_path=path,
                normalize=normalize,
            )

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

    if True:
        # Validate shape of the spectrogram
        dataset_analysis.validate_shape()
