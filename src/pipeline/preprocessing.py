import pandas as pd
import random
import re

def remove_urls(text: str) -> str:
    """Removes URLs from the given text."""
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub('', text)

def normalize_whitespace(text: str) -> str:
    """Normalizes whitespace by replacing multiple spaces with a single space and stripping."""
    return ' '.join(text.split())

def clean_data(df: pd.DataFrame, text_column: str) -> pd.DataFrame:
    """Cleans the dataframe by applying text cleaning rules and dropping duplicates/empty rows."""
    df = df.copy()
    
    # Drop missing values first so they don't become "None" string
    df = df.dropna(subset=[text_column])
    
    # Apply text cleaning
    df[text_column] = df[text_column].astype(str)
    df[text_column] = df[text_column].apply(remove_urls)
    df[text_column] = df[text_column].apply(normalize_whitespace)
    
    # Replace empty strings with NaN and drop missing values again
    df[text_column] = df[text_column].replace('', pd.NA)
    df = df.dropna(subset=[text_column])
    
    # Drop duplicates
    df = df.drop_duplicates(subset=[text_column]).reset_index(drop=True)
    
    return df
