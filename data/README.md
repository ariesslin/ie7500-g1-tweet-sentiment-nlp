# Data Storage and Processing

This project uses the Sentiment140 dataset for sentiment analysis of tweets. Due to the size of the dataset, we employ the following strategy for data storage and processing:

## Raw Data

- The original raw dataset is stored on Google Drive to avoid exceeding GitHub's file size limitations.
- During runtime, the dataset is downloaded into the `data` folder using the `gdown` library.
- The raw dataset is not committed to the GitHub repository to comply with size restrictions and is included in the `.gitignore` file.

## Processed Data

- After preprocessing, the cleaned and processed data is stored in the `processed_data` folder.
- This includes files such as `preprocessed_tweets.csv`, `train_dataset.csv`, `val_dataset.csv`, and `test_dataset.csv`.
- These processed files are included in the repository for easy access and reproducibility of results.

This approach ensures efficient data handling while maintaining compliance with repository size constraints.
