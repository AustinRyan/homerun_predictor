# import requests
# import os
# from dotenv import load_dotenv

# # Load environment variables from .env file
# load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))

# THE_ODDS_API_KEY = os.getenv('THE_ODDS_API_KEY')
# THE_ODDS_API_BASE_URL = 'https://api.the-odds-api.com/v4'

# if not THE_ODDS_API_KEY:
#     print("Warning: THE_ODDS_API_KEY not found in .env file. Odds API calls will fail.")

# def get_sports():
#     """Fetches the list of available sports from The Odds API."""
#     if not THE_ODDS_API_KEY:
#         return {"error": "API key for The Odds API is not configured."}

#     endpoint = f"{THE_ODDS_API_BASE_URL}/sports"
#     params = {
#         'apiKey': THE_ODDS_API_KEY
#     }
#     try:
#         response = requests.get(endpoint, params=params)
#         response.raise_for_status() # Raises an HTTPError for bad responses (4XX or 5XX)
#         return response.json()
#     except requests.exceptions.RequestException as e:
#         print(f"Error fetching sports from The Odds API: {e}")
#         if hasattr(e, 'response') and e.response is not None:
#             print(f"Response content: {e.response.text}")
#         return {"error": str(e)}

# if __name__ == '__main__':
#     print("Fetching available sports from The Odds API...")
#     sports_data = get_sports()
#     if isinstance(sports_data, list):
#         print(f"Successfully fetched {len(sports_data)} sports.")
#         for sport in sports_data:
#             if sport.get('group') == 'Baseball' and sport.get('key') == 'baseball_mlb':
#                 print(f"  - {sport.get('title')} (key: {sport.get('key')}, group: {sport.get('group')}, active: {sport.get('active')})")
#     elif isinstance(sports_data, dict) and 'error' in sports_data:
#         print(f"Error: {sports_data['error']}")
#     else:
#         print("Received unexpected data format for sports.")
#         print(sports_data)
