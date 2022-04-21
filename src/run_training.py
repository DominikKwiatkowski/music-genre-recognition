import gc

from src.data_process.config_paths import DataPathsManager
from src.data_process.metadata_processor import MetadataProcessor
from training.training_config import TrainingConfig
from training.training_manager import run_training

from prepare_data import prepare_data
import multiprocessing

if __name__ == "__main__":
    data_paths = DataPathsManager()
    processor = MetadataProcessor()
    metadata = processor.get_metadata()

    # Prepare data to disk for 3 different types of splits
    for split_id in range(2, 3):
        train, val, test = prepare_data(split_id, data_paths, processor, metadata)

        p = multiprocessing.Process(
            target=run_training,
            args=(
                f"sample-training-{split_id}",
                train,
                data_paths.get_train_dataset_path(split_id),
                val,
                data_paths.get_val_dataset_path(split_id),
                data_paths,
                split_id != 1,
                True,
            ),
        )
        p.start()
        p.join()