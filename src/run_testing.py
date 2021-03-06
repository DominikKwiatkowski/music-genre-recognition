from src.data_process.config_paths import DataPathsManager
from src.data_process.metadata_processor import MetadataProcessor
from testing.testing_manager import test_model

from prepare_data import prepare_data

if __name__ == "__main__":
    data_paths = DataPathsManager()
    processor = MetadataProcessor()
    metadata = processor.get_metadata()

    split_id = 2
    model_id = "99"

    # Prepare data to disk for 3 different types of splits
    model_path = "dumb-spectro192-2"
    train, val, test = prepare_data(split_id, data_paths, processor, metadata)
    test_model(
        model_path,
        model_id,
        data_paths,
        test,
        data_paths.get_test_dataset_path(split_id),
        1,
    )
