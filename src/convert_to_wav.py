import os


def absoluteFilePaths(directory):
    files = []
    for dirpath, _, filenames in os.walk(directory):
        for f in filenames:
            if f.endswith(".mp3"):
                files.append(os.path.abspath(os.path.join(dirpath, f)))
    return files


if __name__ == "__main__":
    files = absoluteFilePaths(r"..\data\fma_small")
    for file in files:
        wav = file.replace("mp3", "wav")
        wav = wav.replace("fma_small", "fma_small_wav")
        wav_dir = os.path.dirname(wav)

        if not os.path.exists(wav_dir):
            os.makedirs(wav_dir)

        if os.path.exists(wav):
            continue

        os.system('ffmpeg -i ' + file + ' ' + wav)
