import os
from typing import List


def get_ext_absolute_filepaths(directory: str, extension: str = "mp3") -> List[str]:
    """
    Returns absolute filepaths of files with given extension in given directory
    :param directory: directory to search in
    :param extension: extension to search for
    :return: list of absolute filepaths
    """
    files = []
    for dirpath, _, filenames in os.walk(directory):
        for f in filenames:
            if f.endswith(extension):
                files.append(os.path.abspath(os.path.join(dirpath, f)))
    return files


def convert_mp3_dataset_to_wav(filepath: str) -> None:
    """
    Converts dataset in mp3 format to wav format.
    Requires ffmpeg to be installed.
    :param filepath: path to dataset
    """

    dataset_name = os.path.basename(filepath)

    files = get_ext_absolute_filepaths(filepath, "mp3")
    for file in files:
        wav = file.replace("mp3", "wav")
        wav = wav.replace(dataset_name, dataset_name + "_wav")
        wav_dir = os.path.dirname(wav)

        if not os.path.exists(wav_dir):
            os.makedirs(wav_dir)

        if os.path.exists(wav):
            continue

        os.system("ffmpeg -i " + file + " " + wav)


if __name__ == "__main__":
    convert_mp3_dataset_to_wav(r"..\data\fma_small")
