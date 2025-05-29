import requests
import datetime
import os
from dotenv import load_dotenv
import pybaseball
import pandas as pd

# Enable pybaseball caching globally for this application's runtime
pybaseball.cache.enable()


# Load environment variables from .env file
# Assuming .env is in the 'backend' directory, so one level up from 'data_fetchers'
dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env')
load_dotenv(dotenv_path=dotenv_path)

MLB_STATS_API_BASE_URL = os.getenv('MLB_STATS_API_BASE_URL', 'https://statsapi.mlb.com/api/v1') # Default if not in .env


def get_mlb_schedule(date_str=None, sport_id=1, include_probable_pitchers=True):
    """Fetches the MLB schedule for a specific date and sportId.

    Args:
        date_str (str, optional): Date in 'YYYY-MM-DD' format. 
                                 If None, fetches today's schedule.
        sport_id (int, optional): The sport ID. Defaults to 1 for MLB.
        include_probable_pitchers (bool, optional): Whether to include probable pitchers
                                  and other detailed information. Defaults to True.

    Returns:
        dict: The JSON response from the API, or None if an error occurs.
    """
    endpoint = f"{MLB_STATS_API_BASE_URL}/schedule"
    
    params = {
        "sportId": sport_id,
    }

    if date_str:
        params["date"] = date_str
    else:
        # Default to today if no date_str is provided
        today = datetime.date.today()
        params["date"] = today.strftime("%Y-%m-%d")
    
    # Add hydration parameters for more detailed information
    if include_probable_pitchers:
        # Add hydrations to get more data including probable pitchers
        params["hydrate"] = "probablePitcher,person,team"

    try:
        print(f"Fetching MLB schedule with params: {params}")
        response = requests.get(endpoint, params=params)
        response.raise_for_status()  # Raise an exception for HTTP errors
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching MLB schedule: {e}")
        return None

def find_player_ids(player_name_query, active=True):
    """Searches for player IDs based on a name query.

    Args:
        player_name_query (str): The name to search for.
        active (bool): Whether to search for active players. True by default.

    Returns:
        list: A list of player dicts matching the query, or None if an error.
              Each dict should at least contain 'id' and 'fullName'.
    """
    # This endpoint is an assumption based on common API patterns and some MLB API examples.
    # The actual endpoint might be /sports/1/players or require different params.
    # The 'toddrob99/MLB-StatsAPI' wrapper uses '/api/v1/sports/{sportId}/players' with params like 'search', 'season'.
    # Let's try a simpler one first: /people/search?names={query}
    # Alternative: /api/v1/sports/1/players?search={query}&season={current_season_if_needed_for_active_status}
    # The official public API might also use different structures.
    # Let's use what statsapi.lookup_player likely uses: https://statsapi.mlb.com/api/v1/people/search

    endpoint = f"{MLB_STATS_API_BASE_URL}/people/search"
    params = {
        "names": player_name_query,
        "active": str(active).lower() # API might expect 'true'/'false' as strings
    }

    try:
        response = requests.get(endpoint, params=params)
        response.raise_for_status()
        data = response.json()
        return data.get('people', []) # The actual key might differ, e.g., 'results', 'players'
    except requests.exceptions.RequestException as e:
        print(f"Error finding player ID for '{player_name_query}': {e}")
        return None
    except KeyError as e:
        print(f"KeyError parsing player search results for '{player_name_query}': {e} - response was {response.text}")
        return None

def get_player_stats(person_id, group, type="season", season=None):
    """Fetches player statistics for a given personId.

    Args:
        person_id (int or str): The player's ID.
        group (str): Stat group, e.g., 'hitting', 'pitching', 'fielding'.
        type (str, optional): Stat type, e.g., 'season', 'career', 'yearByYear'. 
                              Defaults to "season".
        season (str, optional): Season year (e.g., '2023'). Required if type is 'season'.
                                If None and type is 'season', it might default to current 
                                or error.

    Returns:
        dict: The JSON response from the API, or None if an error occurs.
    """
    endpoint = f"{MLB_STATS_API_BASE_URL}/people/{person_id}/stats"
    
    params = {
        "stats": type, # The API seems to use 'stats' for what the wrapper called 'type'
        "group": group,
        "sportId": 1 # Assuming MLB
    }

    if season:
        params["season"] = season
    elif type == "season": # If type is season but no season provided, default to current year
        params["season"] = datetime.date.today().year

    try:
        response = requests.get(endpoint, params=params)
        response.raise_for_status()
        # The stats are usually in a list under a 'splits' key, within 'stats'
        # For example: data['stats'][0]['splits'][0]['stat'] for season stats
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching player stats for ID {person_id}: {e} - URL: {response.url}")
        return None
    except (KeyError, IndexError) as e:
        print(f"Error parsing player stats for ID {person_id}: {e} - response was {response.text}")
        return None


def get_live_game_feed(game_pk):
    """Fetches the live game feed for a specific gamePk, which may contain Statcast data.

    Args:
        game_pk (int or str): The unique ID for the game.

    Returns:
        dict: The JSON response from the API, or None if an error occurs.
    """
    # This endpoint uses API version v1.1
    endpoint = f"https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live"
    
    print(f"Fetching live game feed for gamePk: {game_pk} from {endpoint}")
    try:
        response = requests.get(endpoint)
        response.raise_for_status()  # Raise an exception for HTTP errors
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching live game feed for gamePk {game_pk}: {e}")
        if response is not None:
            print(f"Response status: {response.status_code}, Response text: {response.text[:500]}") # Log part of the response text
        return None
    except Exception as e: # Catch any other unexpected errors, e.g. JSONDecodeError
        print(f"An unexpected error occurred fetching live game feed for gamePk {game_pk}: {e}")
        return None

def get_game_detailed_info(game_pk):
    """Fetches detailed information about a game, including probable pitchers.
    
    This endpoint usually has more detailed information than what's available in the schedule.
    
    Args:
        game_pk (int or str): The unique ID for the game.
        
    Returns:
        dict: The JSON response from the API, or None if an error occurs.
    """
    # This endpoint tends to have more complete information about the game
    endpoint = f"{MLB_STATS_API_BASE_URL}/game/{game_pk}/boxscore"
    
    try:
        print(f"Fetching detailed game info for game {game_pk}")
        response = requests.get(endpoint)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching detailed game info for game {game_pk}: {e}")
        return None

def get_statcast_data_for_date_range(start_date_str, end_date_str=None):
    """ 
    Fetches Statcast data for a given date or date range using pybaseball.
    
    Args:
        start_date_str (str): The start date in 'YYYY-MM-DD' format.
        end_date_str (str, optional): The end date in 'YYYY-MM-DD' format. 
                                      If None, defaults to start_date_str.
                                      
    Returns:
        pandas.DataFrame: A DataFrame containing Statcast data, or None if an error occurs.
    """
    if end_date_str is None:
        end_date_str = start_date_str
        
    # print(f"\n--- Fetching Statcast data using pybaseball for dates: {start_date_str} to {end_date_str} ---") # Commented out for cleaner server logs
    try:
        # Set a higher timeout if needed, default is None (waits indefinitely)
        # For very large date ranges, consider fetching data in smaller chunks (e.g., weekly or monthly)
        statcast_df = pybaseball.statcast(start_dt=start_date_str, end_dt=end_date_str, verbose=False) # Set verbose to False
        if statcast_df is not None and not statcast_df.empty:
            # print(f"Successfully fetched {len(statcast_df)} Statcast events.") # Commented out for cleaner server logs
            # Relevant columns for home run analysis and hit quality:
            relevant_cols = [
                'game_date', 'player_name', 'batter', 'pitcher', 'events', 'description', 'des',
                'zone', 'stand', 'p_throws', 'home_team', 'away_team',
                'type', 'hit_location', 'bb_type', 'balls', 'strikes', 'pfx_x', 'pfx_z',
                'plate_x', 'plate_z', 'on_3b', 'on_2b', 'on_1b',
                'outs_when_up', 'inning', 'inning_topbot', 'hc_x', 'hc_y',
                'fielder_2', 'estimated_ba_using_speedangle', 'estimated_woba_using_speedangle',
                'woba_value', 'woba_denom', 'babip_value', 'iso_value',
                'launch_speed', 'launch_angle', 'launch_speed_angle', 'effective_speed', 'release_spin_rate', 'release_extension',
                'game_pk', 'pitcher_1', 'fielder_2_1', 'fielder_3', 'fielder_4', 'fielder_5',
                'fielder_6', 'fielder_7', 'fielder_8', 'fielder_9',
                'release_speed', 'release_pos_x', 'release_pos_z', 'spin_axis', 'delta_run_exp',
                'home_score', 'away_score', 'bat_score', 'fld_score', 'post_away_score', 'post_home_score',
                'post_bat_score', 'post_fld_score', 'if_fielding_alignment', 'of_fielding_alignment', 'hit_distance_sc'
            ]
            # Filter for columns that actually exist in the returned DataFrame to avoid KeyErrors
            existing_relevant_cols = [col for col in relevant_cols if col in statcast_df.columns]
            
            # Commented out for cleaner server logs:
            # print("Sample of all Statcast data (first 5 rows, all available relevant columns):")
            # pd.set_option('display.max_columns', None) # Show all columns
            # pd.set_option('display.width', 200)      # Adjust display width
            # print(statcast_df[existing_relevant_cols].head())
            
            # Commented out for cleaner server logs:
            # # Specifically look for home runs and their hit data
            # home_runs_df = statcast_df[statcast_df['events'] == 'home_run']
            # if not home_runs_df.empty:
            #     print(f"\nFound {len(home_runs_df)} home runs in the period.")
            #     hr_cols_to_show = ['game_date', 'player_name', 'events', 'launch_speed', 'launch_angle', 'hit_distance_sc', 'des']
            #     existing_hr_cols = [col for col in hr_cols_to_show if col in home_runs_df.columns]
            #     print(home_runs_df[existing_hr_cols].head())
            # else:
            #     print("No home runs found in the fetched data for this period.")
                
            return statcast_df[existing_relevant_cols]
        else:
            # print("No Statcast data returned or DataFrame is empty.") # Commented out for cleaner server logs
            return None
    except Exception as e:
        print(f"Error fetching Statcast data: {e}")
        # Consider more specific error handling if pybaseball throws custom exceptions
        return None

if __name__ == '__main__':
    print("Fetching today's MLB schedule...")
    schedule_today = get_mlb_schedule()
    if schedule_today and schedule_today.get('dates'):
        print(f"Data found for {schedule_today['dates'][0]['date']}:")
        for game in schedule_today['dates'][0]['games']:
            print(f"  {game['teams']['away']['team']['name']} vs {game['teams']['home']['team']['name']} at {game['venue']['name']} ({game['status']['detailedState']})")
    else:
        print("Could not fetch today's schedule or no games today based on response.")
        if schedule_today:
            print("Full response for debugging:", schedule_today) # print response if not as expected

    print("\nFetching schedule for 2024-07-04...") # Example for a specific date
    schedule_specific_date = get_mlb_schedule(date_str="2024-07-04")
    if schedule_specific_date and schedule_specific_date.get('dates'):
        print(f"Data found for {schedule_specific_date['dates'][0]['date']}:")
        for game in schedule_specific_date['dates'][0]['games']:
            print(f"  {game['teams']['away']['team']['name']} vs {game['teams']['home']['team']['name']} at {game['venue']['name']} ({game['status']['detailedState']})")
    else:
        print("Could not fetch schedule for 2024-07-04 or no games on that date based on response.")
        if schedule_specific_date:
             print("Full response for debugging:", schedule_specific_date)

    print("\nSearching for player 'Mike Trout'...")
    trout_search = find_player_ids("Mike Trout")
    if trout_search:
        for player in trout_search:
            print(f"Found: {player.get('fullName')} - ID: {player.get('id')}")
            if player.get('fullName') == "Mike Trout" and player.get('id'): # Be specific
                trout_id = player.get('id')
                print(f"\nFetching season hitting stats for Mike Trout (ID: {trout_id})...")
                trout_hitting_stats = get_player_stats(person_id=trout_id, group='hitting', type='season')
                if trout_hitting_stats and trout_hitting_stats.get('stats'):
                    # The actual stats are nested. Common structure: data['stats'][0]['splits'][0]['stat'] for single season
                    try:
                        season_stats = trout_hitting_stats['stats'][0]['splits'][0]['stat']
                        print(f"  Games Played: {season_stats.get('gamesPlayed')}")
                        print(f"  Home Runs: {season_stats.get('homeRuns')}")
                        print(f"  RBIs: {season_stats.get('rbi')}")
                        print(f"  Batting Average: {season_stats.get('avg')}")
                        # print("Full stats object:", season_stats) # Uncomment to see all returned stats
                    except (IndexError, KeyError) as e:
                        print(f"Could not parse stats structure: {e}")
                        print("Full response for Trout's stats:", trout_hitting_stats)
                else:
                    print("Could not fetch stats for Mike Trout or stats format unexpected.")
                    if trout_hitting_stats:
                        print("Full response for Trout's stats:", trout_hitting_stats)
                break # Found Mike Trout, stop searching
    else:
        print("Player 'Mike Trout' not found or error during search.")

    # Test pybaseball Statcast fetching
    # Use the date of the previously tested completed game
    statcast_game_date = '2024-07-04'
    detailed_statcast_data = get_statcast_data_for_date_range(statcast_game_date)
    # You can further process 'detailed_statcast_data' if needed
    # For example, find specific players or events.

    print("\n--- Testing get_live_game_feed for TODAY's game (original test) ---") # Clarify which test this is
    if schedule_today and schedule_today.get('dates') and schedule_today['dates'][0].get('games'):
        first_game_pk = schedule_today['dates'][0]['games'][0].get('gamePk')
        if first_game_pk:
            print(f"Attempting to fetch live feed for the first game of the day (gamePk: {first_game_pk})...")
            live_feed_data = get_live_game_feed(first_game_pk)
            if live_feed_data:
                print(f"Successfully fetched live game feed for gamePk {first_game_pk}.")
                print(f"Top-level keys in live feed: {list(live_feed_data.keys())}")
                # Example: Check for play-by-play data, which often contains Statcast info
                if 'liveData' in live_feed_data and 'plays' in live_feed_data['liveData'] and 'allPlays' in live_feed_data['liveData']['plays']:
                    print(f"Found 'liveData.plays.allPlays' with {len(live_feed_data['liveData']['plays']['allPlays'])} plays.")
                else:
                    print("Could not find 'liveData.plays.allPlays' in the expected structure.")
                # Avoid printing the whole feed as it's very large
                # print("Full live feed data (first 500 chars):", str(live_feed_data)[:500]) 
            else:
                print(f"Failed to fetch live game feed for gamePk {first_game_pk}.")
        else:
            print("Could not find gamePk for the first game.")
    else:
        print("No games found in today's schedule to test live game feed.")


