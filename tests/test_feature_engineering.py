import pandas as pd
import pytest
from src.feature_engineering import create_lagged_features

def test_create_lagged_features():
    # Sample data
    data = {
        'player_name': ['Player A', 'Player A', 'Player A', 'Player B', 'Player B'],
        'season': [2020, 2021, 2022, 2020, 2021],
        'fantasy_points_ppr': [100, 150, 200, 80, 90],
        'receptions': [10, 15, 20, 5, 7],
        'targets': [20, 25, 30, 10, 12],
        'receiving_yards': [100, 150, 200, 80, 90],
        'reception_td': [1, 2, 3, 0, 1]
    }
    
    df = pd.DataFrame(data)
    
    # Apply the feature engineering function
    df_with_lags = create_lagged_features(df)
    
    # Check if lagged features are created correctly
    assert df_with_lags['fantasy_points_ppr_last_year'].iloc[2] == 150
    assert df_with_lags['receptions_last_year'].iloc[2] == 15
    assert df_with_lags['targets_last_year'].iloc[2] == 25
    assert df_with_lags['receiving_yards_last_year'].iloc[2] == 150
    assert df_with_lags['reception_td_last_year'].iloc[2] == 2

    # Check if the first row for each player has NaN values for lagged features
    assert pd.isna(df_with_lags['fantasy_points_ppr_last_year'].iloc[0])
    assert pd.isna(df_with_lags['receptions_last_year'].iloc[0])