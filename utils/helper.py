import os
import pandas as pd
import gdown
import html

def load_sentiment140(file_path="../data/sentiment140.csv"):
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
