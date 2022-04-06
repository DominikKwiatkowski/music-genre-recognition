import os
import numpy as np
import pandas as pd
import src.data_process.spectrogram_generator as sg


def prepare(dataset_path: str, metadata: pd.DataFrame, save_path: str, normalize: bool):
    """
    Creates spectrogram for all files in the dataset and save to directory.
    """

    # Create save path
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Generate and save spectrogram
    for index in range(len(metadata)):
        file_path = sg.index_to_file_path(dataset_path, metadata, index)
        spectrogram = sg.generate_spectrogram(file_path)
        if normalize:
            spectrogram = sg.normalize_spectrogram(spectrogram)

        filename = str(metadata.iloc[index]["track_id"])
        save_file_path = os.path.join(save_path, filename)
        np.save(save_file_path, spectrogram)
