import pandas as pd

def load_data(filepath):
    """Load the dataset from a CSV file."""
    df = pd.read_csv(filepath)
    return df

def clean_data(df):
    """Clean the dataset by handling missing values and duplicates."""
    # Drop rows with missing fantasy points
    df = df.dropna(subset=['fantasy_points_ppr'])
    
    # Drop duplicates if any
    df = df.drop_duplicates()
    
    return df

def preprocess_data(filepath):
    """Load and preprocess the data."""
    df = load_data(filepath)
    df = clean_data(df)
    return df

