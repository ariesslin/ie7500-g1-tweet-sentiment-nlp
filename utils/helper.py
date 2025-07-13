import os
import pandas as pd
import gdown
import html

def load_sentiment140(file_path="../data/sentiment140.csv"):
    """
    Downloads and loads the Sentiment140 dataset as a DataFrame.

    If the dataset is not already present locally, it will be downloaded from Google Drive.

    Args:
        file_path (str): Path to save or load the Sentiment140 CSV file.

    Returns:
        pd.DataFrame: Raw Sentiment140 dataset with columns:
            ['target', 'id', 'date', 'flag', 'user', 'text']
    """
    url = "https://drive.google.com/uc?id=1OeMI3bTQHZrCchkI-vMv2Ibv6RLGk7aS"
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    if not os.path.exists(file_path):
        gdown.download(url, file_path, quiet=False)

    columns = [
        "target",  # Sentiment (0=negative, 2=neutral, 4=positive)
        "id",      # Tweet ID
        "date",    # Date of tweet
        "flag",    # Query flag
        "user",    # Username
        "text"     # Tweet text
    ]

    df = pd.read_csv(file_path, encoding='latin-1', header=None, names=columns)
    return df



def preprocess_raw_text_in_eda(df, text_col="text"):
    """
    Cleans raw text in a DataFrame for EDA by decoding characters, unescaping HTML,
    removing empty entries, and adding a text length column.

    Args:
        df (pd.DataFrame): DataFrame containing raw tweet data.
        text_col (str): Name of the text column to process.

    Returns:
        pd.DataFrame: Cleaned DataFrame with an added 'text_length' column.
    """
    # Decode and unescape
    df[text_col] = df[text_col].apply(
        lambda x: x.encode("latin1").decode("utf-8", errors="ignore") if isinstance(x, str) else x
    )
    df[text_col] = df[text_col].apply(
        lambda x: html.unescape(x) if isinstance(x, str) else x
    )

    # Remove empty or whitespace-only rows
    df = df[~df[text_col].apply(lambda x: pd.isna(x) or str(x).strip() == '')].copy()

    # Add text length column
    df['text_length'] = df[text_col].apply(len)

    return df



def load_clean_train_val_datasets(train_path="../processed_data/train_dataset.csv", val_path="../processed_data/val_dataset.csv"):
    """
    Loads and prepares the cleaned training and validation datasets.

    Args:
        train_path (str): Path to the training CSV file.
        val_path (str): Path to the validation CSV file.

    Returns:
        train_df (pd.DataFrame): Cleaned training DataFrame with 'text' and 'target'.
        val_df (pd.DataFrame): Cleaned validation DataFrame with 'text' and 'target'.
    """
    # Load CSVs
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)

    # Drop rows with missing text or target
    train_df = train_df.dropna(subset=["text", "target"])
    val_df = val_df.dropna(subset=["text", "target"])

    # Keep only 'text' and 'target' columns
    train_df = train_df[["text", "target"]]
    val_df = val_df[["text", "target"]]

    return train_df, val_df
