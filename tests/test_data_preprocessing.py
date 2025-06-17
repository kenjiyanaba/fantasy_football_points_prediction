import pandas as pd
import pytest
from src.data_preprocessing import load_data, clean_data

def test_load_data():
    df = load_data("data/raw/fantasy_stats.csv")
    assert isinstance(df, pd.DataFrame)
    assert not df.empty

def test_clean_data():
    df = pd.DataFrame({
        'player_name': ['Player A', 'Player B', None],
        'fantasy_points_ppr': [10, None, 15],
        'receptions': [5, 3, None]
    })
    cleaned_df = clean_data(df)
    assert cleaned_df.shape[0] == 2  # Should drop the row with None in player_name
    assert 'player_name' in cleaned_df.columns
    assert 'fantasy_points_ppr' in cleaned_df.columns
    assert 'receptions' in cleaned_df.columns
    assert cleaned_df['fantasy_points_ppr'].isnull().sum() == 0  # No missing values in fantasy_points_ppr
    assert cleaned_df['receptions'].isnull().sum() == 0  # No missing values in receptions