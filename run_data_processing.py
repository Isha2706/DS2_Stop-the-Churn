import os
from src.data.data_processing import main

if __name__ == "__main__":
    # Make sure the telco_train.csv file is in the data/raw directory
    print("Checking for dataset...")
    if not os.path.exists('data/raw/telco_train.csv'):
        print("Warning: telco_train.csv not found in data/raw directory.")
        print("Please place the dataset file in the data/raw directory before running this script.")
    else:
        print("Dataset found. Starting data processing...")
        main()