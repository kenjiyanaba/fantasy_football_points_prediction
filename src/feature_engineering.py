import pandas as pd

def create_lagged_features(df, n_years=3):
    cols_to_shift = ["fantasy_points_ppr"]
    required_cols = ["player_name", "season"] + cols_to_shift
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    df = df.sort_values(by=["player_name", "season"], ascending=True)

    for col in cols_to_shift:
        for i in range(1, n_years + 1):
            df[f"{col}_{i}_year_ago"] = df.groupby("player_name")[col].shift(i)
    return df

def fill_missing_lagged_features(df):
    lagged_cols = [col for col in df.columns if "_year_ago" in col]
    df[lagged_cols] = df[lagged_cols].fillna(0)
    return df

def create_target(df):
    # Predict next year's fantasy points
    df['target_fantasy_points_ppr'] = df.groupby("player_name")["fantasy_points_ppr"].shift(-1)
    df = df[df['target_fantasy_points_ppr'].notna()]  # Drop rows without a target
    return df

def feature_engineering_pipeline(df):
    df = create_lagged_features(df)
    df = fill_missing_lagged_features(df)
    df = create_target(df)  # Add this line
    return df
