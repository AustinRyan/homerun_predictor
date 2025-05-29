# # backend/prepare_features.py
# import pandas as pd
# import numpy as np
# import os

# # Ensure the data directory exists for output
# if not os.path.exists('data'):
#     os.makedirs('data')

# def prepare_statcast_features(input_csv: str = "data/all_statcast.csv", output_csv: str = "data/features.csv"):
#     """Reads combined Statcast data, engineers features, and saves the result.

#     Args:
#         input_csv: Path to the combined Statcast CSV file.
#         output_csv: Path to save the feature-engineered CSV file.
#     """
#     print(f"Reading data from {input_csv}...")
#     try:
#         df = pd.read_csv(input_csv, low_memory=False) # low_memory=False to handle mixed types if any
#     except FileNotFoundError:
#         print(f"Error: Input file {input_csv} not found. Please run fetch_data.py first.")
#         return

#     print(f"Initial rows: {len(df)}")

#     # Filter to rows where 'events' is not null (completed PAs)
#     df.dropna(subset=['events'], inplace=True)
#     print(f"Rows after filtering for non-null 'events': {len(df)}")

#     # Create binary target column
#     df["is_hr"] = (df["events"] == "home_run").astype(int)

#     # Feature Engineering
#     # Ensure required columns exist, fill NaNs with a neutral value or drop
#     # For numeric features used in models, NaN handling is crucial.
#     # For simplicity, we'll fill with 0 or median, but more sophisticated imputation might be needed.
    
#     # Launch Speed (launch_speed)
#     df['launch_speed'] = pd.to_numeric(df['launch_speed'], errors='coerce').fillna(0)
    
#     # Launch Angle (launch_angle)
#     df['launch_angle'] = pd.to_numeric(df['launch_angle'], errors='coerce').fillna(0)
    
#     # Ideal Launch Angle (binary: 20 <= launch_angle <= 40)
#     df['ideal_launch_angle'] = ((df['launch_angle'] >= 20) & (df['launch_angle'] <= 40)).astype(int)
    
#     # Hard Hit (binary: launch_speed >= 95 mph)
#     df['hard_hit'] = (df['launch_speed'] >= 95).astype(int)
    
#     # Estimated wOBA using speed and angle (estimated_woba_using_speedangle)
#     df['estimated_woba_using_speedangle'] = pd.to_numeric(df['estimated_woba_using_speedangle'], errors='coerce').fillna(0)
    
#     # Hit Distance (hit_distance_sc)
#     df['hit_distance_sc'] = pd.to_numeric(df['hit_distance_sc'], errors='coerce').fillna(0)

#     # Select features and target
#     # These are the features the model will be trained on, based on the prompt.
#     feature_columns = [
#         'launch_speed',
#         'launch_angle',
#         'ideal_launch_angle',
#         'hard_hit',
#         'estimated_woba_using_speedangle',
#         'hit_distance_sc'
#     ]
#     target_column = 'is_hr'

#     # Keep only selected features and the target variable
#     # Also, it's good practice to keep identifiers if needed for later analysis, e.g., game_pk, at_bat_number, pitcher, batter
#     # For now, strictly adhering to features for model training as per prompt.
#     final_df = df[feature_columns + [target_column]].copy()
    
#     # Handle any remaining NaNs in feature columns specifically (e.g., if original columns were all NaNs)
#     final_df.fillna(0, inplace=True) # A simple fillna, consider more robust methods for production

#     print(f"Saving feature matrix with {len(final_df)} rows and target to {output_csv}")
#     final_df.to_csv(output_csv, index=False)
#     print("Feature engineering and labeling process finished.")

# if __name__ == "__main__":
#     prepare_statcast_features()
