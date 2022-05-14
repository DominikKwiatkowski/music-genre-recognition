import os
from typing import Tuple
import pandas as pd

from src.data_process.config_paths import DataPathsManager
from src.data_process.metadata_processor import MetadataProcessor
from src.data_process import spectrogram_generator


def prepare_data(
    split_id: int,
    data_paths: DataPathsManager,
    metadata_processor: MetadataProcessor,
    metadata: pd.DataFrame,
    show_plots: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    For given split type, split and save data to disk.
    :param split_id: split type
    :param data_paths: DataPathsManager object
    :param metadata_processor: MetadataProcessor object
    :param metadata: metadata dataframe
    :param show_plots: whether to show plots of dataset distribution
    :return: None
    """
    if split_id == 1:
        train, val, test = metadata_processor.split_metadata(metadata, 0.8, 0.1, 0.1)
        normalize = False
    elif split_id == 2:
        train, val, test = metadata_processor.split_metadata_uniform(
            metadata, 0.8, 0.1, 0.1
        )
        normalize = True
    elif split_id == 3:
        train, val, test = metadata_processor.split_metadata_uniform(
            metadata, 0.8, 0.1, 0.1, add_val_to_train=True
        )
        normalize = True
    else:
        raise ValueError("Invalid split_id. It should be 1, 2 or 3.")

    if show_plots:
        MetadataProcessor.plot_genres_distribution(metadata, "Metadata distribution")
        MetadataProcessor.plot_genres_distribution(train, "TRAIN set distribution")
        MetadataProcessor.plot_genres_distribution(val, "VAL set distribution")
        MetadataProcessor.plot_genres_distribution(test, "TEST set distribution")
        MetadataProcessor.plot_sizes(train, val, test)

    # Prepare dataset and generate spectrogram
    for dataset_metadata, path in (
        (train, data_paths.get_train_dataset_path(split_id)),
        (val, data_paths.get_val_dataset_path(split_id)),
        (test, data_paths.get_test_dataset_path(split_id)),
    ):
        if not os.path.exists(path):
            spectrogram_generator.generate_all_spectrograms(
                dataset_path=data_paths.datasetPath,
                metadata=dataset_metadata,
                save_path=path,
                normalize=normalize,
            )

    return train, val, test


if __name__ == "__main__":
    _data_paths = DataPathsManager()
    _processor = MetadataProcessor()
    _metadata: pd.DataFrame = _processor.get_metadata()

    # Prepare data to disk for 3 different types of splits
    for _split_id in range(2, 3):
        prepare_data(_split_id, _data_paths, _processor, _metadata)
