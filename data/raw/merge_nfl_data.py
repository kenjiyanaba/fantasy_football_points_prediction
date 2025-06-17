import pandas as pd

# Load your main NFL stats CSV
main_df = pd.read_csv('/Users/kenjiyanaba/Python/NFLScraper/fantasy-points-prediction/data/raw/fantasy_stats.csv')

# Load the grades CSV
grades_df = pd.read_csv('/Users/kenjiyanaba/Python/NFLScraper/fantasy-points-prediction/data/raw/Merge_this.csv')

# Merge the two DataFrames on player_name
# If season matters, you can add 'season' to the list of keys
merged_df = pd.merge(main_df, grades_df, on='player_name', how='left')

# Save the result to a new CSV
merged_df.to_csv('merged_nfl_data.csv', index=False)

print("Merged CSV saved as 'merged_nfl_data.csv'")





# # Load the CSV
# df = pd.read_csv('/Users/kenjiyanaba/Python/NFLScraper/fantasy-points-prediction/data/raw/nfl_draft_prospects_full.csv')

# # Drop the 'prospect_grades' column
# df = df.drop(columns=['draft_year', 'position'])

# # Save the updated CSV (overwrite original or save as new)
# df.to_csv('Merge_this.csv', index=False)

# print("Column 'prospect_grades' deleted successfully.")