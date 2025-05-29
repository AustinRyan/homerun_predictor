# # backend/fetch_data.py
# import pandas as pd
# from pybaseball import statcast
# import os
# from datetime import datetime

# # Ensure the data directory exists
# if not os.path.exists('data'):
#     os.makedirs('data')

# def fetch_statcast_data(start_date: str, end_date: str, out_csv: str) -> pd.DataFrame:
#     """Fetches Statcast data for a given date range and saves it to a CSV file.

#     Args:
#         start_date: The start date for data fetching (YYYY-MM-DD).
#         end_date: The end date for data fetching (YYYY-MM-DD).
#         out_csv: The path to the output CSV file.

#     Returns:
#         A pandas DataFrame containing the fetched Statcast data.
#     """
#     print(f"Fetching data from {start_date} to {end_date}...")
#     try:
#         data = statcast(start_dt=start_date, end_dt=end_date, verbose=False)
#         if not data.empty:
#             print(f"Successfully fetched {len(data)} rows.")
#             data.to_csv(out_csv, index=False)
#             print(f"Saved data to {out_csv}")
#             return data
#         else:
#             print(f"No data found for {start_date} to {end_date}.")
#             return pd.DataFrame()
#     except Exception as e:
#         print(f"Error fetching data for {start_date}-{end_date}: {e}")
#         return pd.DataFrame()

# if __name__ == "__main__":
#     all_data_frames = []
#     current_year = datetime.now().year

#     # Loop through seasons 2017 up to and including the current year (or 2024 if current year is beyond)
#     # Statcast data typically starts becoming robust around March for a given year.
#     # Regular season typically ends around start of October.
#     for year in range(2017, min(current_year + 1, 2025)):
#         start_date = f"{year}-03-01" # Broad start to catch early games
#         end_date = f"{year}-11-01"   # Broad end to catch late games/post-season
        
#         # Adjust for the current year if it's not complete
#         if year == current_year:
#             today = datetime.now().strftime('%Y-%m-%d')
#             if today < end_date:
#                  end_date = today # Fetch up to yesterday if current year
#                  if start_date > end_date: # Ensure start_date is not after end_date for ongoing season
#                      print(f"Skipping year {year} as start date {start_date} is after current processing date {end_date}.")
#                      continue

#         # For the purpose of the prompt, using the specified date ranges if fixed historical pull is intended
#         # For a more dynamic pull, the logic above is more robust.
#         # Prompt specified: "YYYY-03-28":"YYYY-10-01"
#         prompt_start_date = f"{year}-03-28"
#         prompt_end_date = f"{year}-10-01"

#         file_name = f"data/statcast_{year}.csv"
        
#         # Check if file already exists to avoid re-downloading
#         if os.path.exists(file_name):
#             print(f"Data for {year} already exists at {file_name}. Loading from file.")
#             year_data = pd.read_csv(file_name)
#         else:
#             # Using prompt-specified dates; adjust if more dynamic fetching is desired.
#             year_data = fetch_statcast_data(prompt_start_date, prompt_end_date, file_name)
        
#         if not year_data.empty:
#             all_data_frames.append(year_data)

#     if all_data_frames:
#         print("Concatenating all yearly dataframes...")
#         combined_data = pd.concat(all_data_frames, ignore_index=True)
#         combined_data.to_csv("data/all_statcast.csv", index=False)
#         print(f"Saved combined Statcast data to data/all_statcast.csv")
#         print(f"Total number of rows fetched and combined: {len(combined_data)}")
#     else:
#         print("No data was fetched or loaded. 'data/all_statcast.csv' will not be created.")

#     print("Data acquisition process finished.")
