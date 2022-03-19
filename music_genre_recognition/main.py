from data_preprocessor import DataPreprocessor

if __name__ == "__main__":
    preprocessor = DataPreprocessor()
    # preprocessor.create_data_set_csv()
    preprocessor.load_data()
    # preprocessor.plot_data()
    preprocessor.create_spectograms()
