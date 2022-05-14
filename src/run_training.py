import gc

from src.data_process.config_paths import DataPathsManager
from src.data_process.metadata_processor import MetadataProcessor
from training.training_config import TrainingConfig
from training.training_manager import run_training
from training.training_manager import run_training_new

from prepare_data import prepare_data
import multiprocessing

if __name__ == "__main__":
    data_paths = DataPathsManager()
    processor = MetadataProcessor()
    metadata = processor.get_metadata()

    # Prepare data to disk for 3 different types of splits
    for split_id in range(2, 3):
        train, val, test = prepare_data(split_id, data_paths, processor, metadata)

        run_training(
            f"sample-training-larger-model-{split_id}",
            train,
            data_paths.get_train_dataset_path(split_id),
            val,
            data_paths.get_val_dataset_path(split_id),
            data_paths,
            augment=True,
            overwrite_previous=True
        )

        # run_training_new(
        #     f"sample-training-changed-augment-{split_id}",
        #     train,
        #     data_paths.get_train_dataset_path(split_id),
        #     val,
        #     data_paths.get_val_dataset_path(split_id),
        #     data_paths,
        #     augment=True,
        #     overwrite_previous=True
        # )

        # p = multiprocessing.Process(
        #     target=run_training,
        #     args=(
        #         f"sample-training-fixed-lr-{split_id}",
        #         train,
        #         data_paths.get_train_dataset_path(split_id),
        #         val,
        #         data_paths.get_val_dataset_path(split_id),
        #         data_paths,
        #         split_id != 1,
        #         True,
        #     ),
        # )
        # p.start()
        # p.join()
