import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from feature_engineering import feature_engineering_pipeline
import matplotlib.pyplot as plt
import logging
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import randint
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.svm import SVR
from catboost import CatBoostRegressor



# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the preprocessed data
filepath = "/Users/kenjiyanaba/Python/NFLScraper/fantasy-points-prediction/data/raw/fantasy_stats.csv"
df = pd.read_csv(filepath)
# Drop rows where 'grade' is missing
df = df.dropna(subset=["grade"])

# Apply feature engineering pipeline
df = feature_engineering_pipeline(df)

# List of positions to test
positions = ["QB", "RB", "WR", "TE"]
universal_features = ["fantasy_points_ppr_1_year_ago", "fantasy_points_ppr_2_year_ago", "fantasy_points_ppr_3_year_ago",
                     "age", "season", "grade", "games", "ppg", "ypg", "offense_pct", "offense_snaps", "team_total_snaps", "td_pct", "total_tds", "touches", "total_yards",
                     "injuries", "career_injuries",
                     "seasons_played", "nfl_years"]


# Define position-specific features
position_features = {
    "QB": universal_features + ["pass_attempts", "complete_pass", "passing_yards", "pass_td", "interception", "ypa", "pass_td_pct", "int_pct", "passer_rating", 
                                "career_pass_attempts", "career_pass_td", "career_interception", "average_ypa", "team_pass_snaps_pct", "team_pass_attempts", 
                                "vacated_pass_attempts", "vacated_pass_td"],
    "RB": universal_features + ["rush_attempts", "rushing_yards", "run_td", "ypc", "receptions", "receiving_yards", "yards_after_catch", "target_share",
                                "team_rush_snaps_pct", "team_rushing_yards", "team_run_td", "vacated_rush_attempts", "vacated_rushing_yards"],
    "WR": universal_features + ["targets", "receptions", "receiving_yards", "receiving_air_yards", "yards_after_catch", "reception_td", "air_yards_share", "target_share", 
                                "yptarget", "ypr", "rec_td_pct", "team_targets", "team_receptions", "team_receiving_yards", "vacated_targets", "vacated_receiving_yards"],
    "TE": universal_features + ["targets", "receptions", "receiving_yards", "receiving_air_yards", "yards_after_catch", "reception_td", "air_yards_share", "target_share", 
                                "yptarget", "ypr", "rec_td_pct", "team_targets", "team_receptions", "team_receiving_yards", "vacated_targets", "vacated_receiving_yards"]
}

# Impute vacated features with 0
vacated_features = [
    "vacated_pass_attempts", "vacated_pass_td",
    "vacated_rush_attempts", "vacated_rushing_yards",
    "vacated_targets", "vacated_receiving_yards"
]

df[vacated_features] = df[vacated_features].fillna(0)



for position in positions:
    print(f"\nTraining model for position: {position}")
    
    # Filter data for the current position
    df_position = df[df["position"] == position]
    
    # Define features and target
    features = position_features.get(position, [])
    target = "fantasy_points_ppr"
    
    # Ensure the filtered data has the required columns
    if df_position[features].isnull().any().any():
        print(f"Skipping position {position} due to missing data in features.")
        continue
    
    X = df_position[features]
    y = df_position[target]
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Define the model
    model = GradientBoostingRegressor(random_state=42)

    # Define hyperparameter grid
    param_grid = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }





    random_search = RandomizedSearchCV(
        model, param_distributions=param_grid, 
        n_iter=50, cv=3, scoring='neg_mean_squared_error', n_jobs=-1, random_state=42
    )
    random_search.fit(X_train, y_train)
    best_model = random_search.best_estimator_






    # Define hyperparameter grid
    # param_grid = {
    #     'n_estimators': [100, 200, 300, 500],
    #     'max_depth': [None, 10, 20, 30],
    #     'min_samples_split': [2, 5, 10],
    #     'min_samples_leaf': [1, 2, 4]
    # }
    
    # # Perform Grid Search
    # grid_search = GridSearchCV(model, param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
    # grid_search.fit(X_train, y_train)
    
    # # Get the best model
    # best_model = grid_search.best_estimator_

    # Old one: Analyze feature importance
    # feature_importances = best_model.feature_importances_
    # importance_dict = dict(zip(features, feature_importances))
    # sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)

    # print(f"Feature Importance for {position}:")
    # for feature, importance in sorted_importance:
    #     print(f"{feature}: {importance:.4f}")





    feature_importances = best_model.feature_importances_
    features = X_train.columns
    importance_df = pd.DataFrame({
        'feature': features,
        'importance': feature_importances
    }).sort_values(by='importance', ascending=False)

    importance_df.plot(kind='barh', x='feature', y='importance', figsize=(10, 6))
    plt.title('Feature Importance')
    plt.gca().invert_yaxis()
    plt.show()

    
    # Make predictions
    y_pred = best_model.predict(X_test)
    
    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    print(f"Position: {position}")
    print(f"Mean Squared Error: {mse}")
    print(f"R^2 Score: {r2}")
    print(f"MAE: {mae}")
    
    # Save the best model for the current position
    model_path = f"/Users/kenjiyanaba/Python/NFLScraper/fantasy-points-prediction/models/random_forest_model_{position}.pkl"
    joblib.dump(best_model, model_path)
    print(f"Best model saved for position {position} at {model_path}")
    