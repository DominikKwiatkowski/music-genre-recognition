import os
import urllib.request

from alive_progress import alive_bar


class DownloadProgressBar:
    def __init__(self):
        self.bar = None
        self.updateMethod = None

    def __call__(self, block_num, block_size, total_size):
        if not self.bar:
            self.bar = alive_bar(total_size, title=f"Downloading status")
            self.updateMethod = self.bar.__enter__()

        self.updateMethod(block_size)


def unzip(filepath: str):
    """
    Unzip file with zipfile.
    """

    import zipfile

    with zipfile.ZipFile(filepath, "r") as zip_ref:
        zip_ref.extractall(os.path.dirname(filepath))


def download_data(target_dir: str, url: str):
    """
    Download and unzip data.
    """

    data_name = url.split("/")[-1]

    # Download zip from url and save it to target_dir as name from url
    print(f"Downloading: {url}")
    urllib.request.urlretrieve(url, f"{target_dir}/{data_name}", DownloadProgressBar())

    # Unzip data with multiple threads
    print(f"Unzipping data: {target_dir}/{data_name}")
    os.system(f"unzip {target_dir}/{data_name} -d {target_dir}")

    # Remove zip file
    print(f"Removing zip file: {target_dir}/{data_name}")
    os.remove(f"{target_dir}/{data_name}")


def setup_environment(url: str):
    """
    Download and unpack data.
    """

    target_dir = os.path.dirname(os.path.realpath(__file__)) + "/../data"

    # Create target directory
    if not os.path.exists(target_dir):
        os.mkdir(target_dir)

    # Validate if url finishes with .zip
    if not url.endswith(".zip"):
        print("URL must end with .zip")
        exit(1)

    download_data(target_dir, url)


if __name__ == "__main__":
    # Parse arguments with argparse
    import argparse

    parser = argparse.ArgumentParser(description="Download and unpack data.")
    parser.add_argument("url", type=str, help="URL to download")
    args = parser.parse_args()

    setup_environment(args.url)
