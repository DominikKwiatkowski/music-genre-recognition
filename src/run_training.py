import gc

from src.data_process.config_paths import DataPathsManager
from src.data_process.metadata_processor import MetadataProcessor
from training.training_config import TrainingConfig
from training.training_manager import run_training

from prepare_data import prepare_data

if __name__ == "__main__":
    data_paths = DataPathsManager()
    processor = MetadataProcessor()
    metadata = processor.get_metadata()

    # Prepare data to disk for 3 different types of splits
    for split_id in range(1, 4):
        train, val, test = prepare_data(split_id, data_paths, processor, metadata)

        default_config = TrainingConfig()
        run_training(
            f"sample-training-{split_id}",
            default_config,
            train,
            val,
            data_paths,
            split_id,
            overwrite_previous=True
        )
