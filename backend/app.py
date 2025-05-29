from flask import Flask, jsonify, request
import datetime
from data_fetchers.weather_api import get_weather_forecast
from data_fetchers.mlb_api import get_mlb_schedule, find_player_ids, get_player_stats, get_statcast_data_for_date_range, get_game_detailed_info
from algorithm.predictor import (
    calculate_batter_overall_stats,
    calculate_pitcher_overall_stats,
    calculate_batter_handedness_stats,
    calculate_pitcher_handedness_stats,
    generate_hr_likelihood_score
)
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import os
import json
import requests
import json # Added for debug printing
import numpy as np

# Custom JSON encoder to handle NumPy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)
import io
import sys
from contextlib import redirect_stdout
# from flask_cors import CORS

# Custom class to capture print statements while still printing to console
class TeeIO(io.StringIO):
    def __init__(self, original_stdout):
        super().__init__()
        self.original_stdout = original_stdout
    
    def write(self, text):
        # Write to the original stdout (console)
        self.original_stdout.write(text)
        # Also write to the StringIO buffer
        return super().write(text)
    
    def flush(self):
        self.original_stdout.flush()
        return super().flush()

app = Flask(__name__)
app.json_encoder = NumpyEncoder  # Use our custom encoder for JSON responses

# Configure CORS to allow requests from Vercel frontend
# CORS(app, resources={r"/*": {"origins": ["https://homerun-predictor.vercel.app", "http://localhost:3000", "http://localhost:5173"]}})  # Enable CORS for specific origins

def make_json_serializable(data):
    if isinstance(data, dict):
        return {k: make_json_serializable(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [make_json_serializable(i) for i in data]
    elif isinstance(data, (np.integer, np.int64)):
        return int(data)
    elif isinstance(data, (np.floating, np.float64)):
        return float(data)
    elif isinstance(data, pd.Timestamp):
        return data.isoformat()
    elif pd.isna(data):
        return None 
    return data


@app.route('/')
def home():
    return jsonify(message="Welcome to the Home Run Parlay Model API!")

@app.route('/api/test')
def test_endpoint():
    return jsonify(status="API is running")

# We will add more endpoints for games, players, pitchers, predictions etc.

@app.route('/api/weather_test')
def weather_test_endpoint():
    # Example: New York City coordinates
    # In a real scenario, these would come from game data or user input
    lat = request.args.get('lat', default=40.7128, type=float)
    lon = request.args.get('lon', default=-74.0060, type=float)
    
    weather_data = get_weather_forecast(lat, lon)
    if weather_data:
        return jsonify(weather_data)
    else:
        return jsonify(error="Failed to fetch weather data"), 500

@app.route('/api/mlb_schedule_test')
def mlb_schedule_test_endpoint():
    date_param = request.args.get('date') # Expects YYYY-MM-DD or None for today
    schedule_data = get_mlb_schedule(date_str=date_param)
    if schedule_data:
        return jsonify(schedule_data)
    else:
        return jsonify(error="Failed to fetch MLB schedule"), 500

@app.route('/api/mlb_find_player')
def mlb_find_player_endpoint():
    player_name = request.args.get('name')
    if not player_name:
        return jsonify(error="'name' query parameter is required"), 400
    
    player_data = find_player_ids(player_name)
    if player_data is not None: # find_player_ids returns [] for no matches, None for error
        return jsonify(player_data)
    else:
        return jsonify(error=f"Failed to find player or error in API for '{player_name}'"), 500

@app.route('/api/mlb_player_stats')
def mlb_player_stats_endpoint():
    player_id = request.args.get('player_id')
    group = request.args.get('group') # 'hitting', 'pitching', 'fielding'
    stat_type = request.args.get('type', default='season') # 'season', 'career', 'yearByYear'
    season = request.args.get('season') # e.g., '2023'

    if not player_id or not group:
        return jsonify(error="'player_id' and 'group' query parameters are required"), 400
    
    # Validate group
    if group not in ['hitting', 'pitching', 'fielding']:
        return jsonify(error="'group' must be one of 'hitting', 'pitching', 'fielding'"), 400

    # Validate type
    if stat_type not in ['season', 'career', 'yearByYear']:
        return jsonify(error="'type' must be one of 'season', 'career', 'yearByYear'"), 400

    stats_data = get_player_stats(person_id=player_id, group=group, type=stat_type, season=season)
    
    if stats_data:
        return jsonify(stats_data)
    else:
        return jsonify(error=f"Failed to fetch player stats for ID {player_id}"), 500

@app.route('/api/odds_sports')
def odds_sports_endpoint():
    sports_data = get_odds_api_sports()
    if isinstance(sports_data, list): # Successful response is a list
        return jsonify(sports_data)
    elif isinstance(sports_data, dict) and 'error' in sports_data:
        return jsonify(sports_data), 500 # Pass through error from the fetcher
    else:
        return jsonify(error="Failed to fetch sports from The Odds API or unexpected format"), 500

@app.route('/api/statcast/<string:date_str>')
def statcast_date_endpoint(date_str):
    """Fetches Statcast data for a specific date."""
    # Basic validation for date_str format (YYYY-MM-DD)
    try:
        datetime.datetime.strptime(date_str, '%Y-%m-%d')
    except ValueError:
        return jsonify(error="Invalid date format. Please use YYYY-MM-DD."), 400

    statcast_df = get_statcast_data_for_date_range(start_date_str=date_str)
    
    if statcast_df is not None and not statcast_df.empty:
        # Convert DataFrame to JSON. orient='records' creates a list of dicts.
        return jsonify(statcast_df.to_json(orient='records'))
    elif statcast_df is not None and statcast_df.empty:
        return jsonify(message=f"No Statcast data found for {date_str}", data=[]) # Return empty list for no data
    else:
        return jsonify(error=f"Failed to fetch Statcast data for {date_str}"), 500


@app.route('/api/homerun_predictions/<string:date_str>', methods=['POST', 'OPTIONS'], defaults={'limit': None})
@app.route('/api/homerun_predictions/<string:date_str>/<int:limit>', methods=['POST', 'OPTIONS'])
def homerun_predictions_endpoint(date_str, limit=None):
    if request.method == 'OPTIONS':
        return '', 200 # Flask-CORS should handle headers, we just prevent further execution

    # Create a log capture object that also writes to the original stdout
    old_stdout = sys.stdout
    log_capture = TeeIO(old_stdout)
    sys.stdout = log_capture
    
    # Add a test log to verify logging is working
    print(f"Starting homerun prediction calculation for date {date_str} with limit {limit}")

    # --- Configuration Parameters --- 
    # Defaults will be used if not provided in POST request or if it's a GET request
    config = {}
    if request.method == 'POST' and request.is_json:
        config = request.json
        print(f"Received POST request with JSON config: {config}")
    else:
        print("No JSON config in POST or it's a GET request, using defaults.")

    # Extract configuration from request.json, with defaults from predictor.py or function signatures
    config_data = request.json if request.method == 'POST' and request.json else {}
    
    try:
        current_date_obj = datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        return jsonify({"error": "Invalid date format. Please use YYYY-MM-DD."}), 400

    # Determine the date range for Statcast data fetching
    today_actual = datetime.now().date()
    if current_date_obj == today_actual:
        end_date_fetch = current_date_obj - timedelta(days=1)
        print(f"Request for today's date ({date_str}). Fetching Statcast data up to yesterday: {end_date_fetch.strftime('%Y-%m-%d')}")
    else:
        end_date_fetch = current_date_obj
        print(f"Request for date {date_str}. Fetching Statcast data up to: {end_date_fetch.strftime('%Y-%m-%d')}")
    
    start_date_fetch = datetime(end_date_fetch.year, 1, 1).date() # Start of the year for end_date_fetch
    print(f"Statcast data window: {start_date_fetch.strftime('%Y-%m-%d')} to {end_date_fetch.strftime('%Y-%m-%d')}")

    statcast_df_period = get_statcast_data_for_date_range(
        start_date_fetch.strftime("%Y-%m-%d"),
        end_date_fetch.strftime("%Y-%m-%d")
    )

    if statcast_df_period is None or statcast_df_period.empty:
        return jsonify({"error": f"Could not fetch Statcast data for the period {start_date_fetch.strftime('%Y-%m-%d')} to {end_date_fetch.strftime('%Y-%m-%d')}."}), 500

    for col in ['batter', 'pitcher']:
        if col in statcast_df_period.columns:
            statcast_df_period[col] = pd.to_numeric(statcast_df_period[col], errors='coerce').astype('Int64')

    print(f"Fetching MLB schedule for {date_str} to identify matchups...")
    schedule_data = get_mlb_schedule(date_str)
    
     # Define a dict to store ballpark factors for each venue (name-keyed)
    # Factors are illustrative and should be tuned based on data.
    # Using names makes it easier to map from game data.
    ballpark_hr_factors = {
        # American League
        "Yankee Stadium": 1.15,
        "Fenway Park": 0.90,
        "Oriole Park at Camden Yards": 0.85,
        "Tropicana Field": 0.95,  # Corrected Rays' stadium, adjusted factor from 1.10 to 0.95 (pitcher-friendly)
        "Rogers Centre": 1.00,
        "Guaranteed Rate Field": 1.05,
        "Progressive Field": 0.85,
        "Comerica Park": 0.90,
        "Kauffman Stadium": 0.95,
        "Target Field": 1.05,
        "Daikin Park": 1.10,        # User-specified name for Astros' park (was Minute Maid Park), kept factor 1.10
        "Sutter Health Park": 1.45, # User-specified name for Athletics' temp park, kept factor 1.20
        "T-Mobile Park": 0.85,
        "Globe Life Field": 1.25,
        # National League
        "Wrigley Field": 0.85,
        "Great American Ball Park": 1.25,
        "PNC Park": 0.80,
        "American Family Field": 1.10,
        "Busch Stadium": 0.90,
        "Coors Field": 1.45,
        "Dodger Stadium": 0.95,
        "Oracle Park": 0.80,
        "Petco Park": 0.95,
        "Chase Field": 1.10,
        "Citi Field": 1.00,
        "Citizens Bank Park": 1.15,
        "Nationals Park": 0.95,
        "loanDepot park": 0.90, # Formerly Marlins Park
        "Truist Park": 1.00,
        # Add any other parks if necessary, e.g., for London Series, Mexico City Series if they have specific factors
        "Estadio Alfredo Harp Helú": 1.20, # Example for Mexico City games
        "London Stadium": 1.10, # Example for London games
    }

    # Default factor for parks not listed or if venue name is missing/mismatched
    DEFAULT_BALLPARK_FACTOR = 1.0
    
    # Initialize dictionary to store weather data by game
    games_weather_data = {}
    
    if not schedule_data or not schedule_data.get("dates") or not schedule_data["dates"][0].get("games"):
        return jsonify({"message": f"Could not fetch MLB schedule for {date_str} or no games scheduled."}), 200

    # Dictionary to store weather data for each game
    games_weather_data = {}
    
    # Print raw schedule data for debugging
    print(f"Schedule data keys: {schedule_data.keys()}")
    if 'dates' in schedule_data and schedule_data['dates']:
        print(f"Number of game dates: {len(schedule_data['dates'])}")
        for date_idx, date_data in enumerate(schedule_data['dates']):
            print(f"Date {date_idx}: {date_data.get('date', 'Unknown date')}")
            if 'games' in date_data:
                print(f"  Number of games for this date: {len(date_data['games'])}")
                # Print structure of first game to understand the format
                if date_data['games']:
                    first_game = date_data['games'][0]
                    print(f"  First game keys: {first_game.keys()}")
                    
                    # Check for game link
                    if 'link' in first_game:
                        print(f"  Game link found: {first_game['link']}")
                    if 'gamePk' in first_game:
                        print(f"  Game ID found: {first_game['gamePk']}")
                        
                    if 'teams' in first_game:
                        print(f"  Teams section keys: {first_game['teams'].keys()}")
                        # Check if probablePitcher exists for home and away teams
                        if 'home' in first_game['teams']:
                            print(f"  Home team keys: {first_game['teams']['home'].keys()}")
                            if 'probablePitcher' in first_game['teams']['home']:
                                print(f"  Home probable pitcher: {first_game['teams']['home']['probablePitcher']}")
                            else:
                                print("  No probablePitcher key found for home team")
                        if 'away' in first_game['teams']:
                            print(f"  Away team keys: {first_game['teams']['away'].keys()}")
                            if 'probablePitcher' in first_game['teams']['away']:
                                print(f"  Away probable pitcher: {first_game['teams']['away']['probablePitcher']}")
                            else:
                                print("  No probablePitcher key found for away team")
    
    # Extract game info including venue and weather data
    game_data = {} # Initialize as a dictionary
    all_scheduled_player_ids = set() # Initialize set for all players scheduled for the day
    player_id_to_name_map = {} # Initialize map for player IDs to names
    for game in schedule_data["dates"][0].get("games", []):
        game_pk = game.get("gamePk")
        current_game_home_player_ids = set() # Initialize at loop start
        current_game_away_player_ids = set() # Initialize at loop start
        if not game_pk:
            continue

        # Fetch boxscore for player lists for this game
        if game_pk: # This check is a bit redundant given the continue above, but safe
            # print(f"Fetching boxscore for game_pk: {game_pk} to get player rosters.") # Verbose, can enable if needed
            detailed_game_info_for_roster = get_game_detailed_info(game_pk) # This is the boxscore data

            if detailed_game_info_for_roster and 'teams' in detailed_game_info_for_roster:
                for team_type in ['home', 'away']:
                    if team_type in detailed_game_info_for_roster['teams'] and detailed_game_info_for_roster['teams'][team_type]:
                        team_data = detailed_game_info_for_roster['teams'][team_type]
                        # Add batters (starting lineup)
                        if 'batters' in team_data and isinstance(team_data['batters'], list):
                            all_scheduled_player_ids.update(player_id for player_id in team_data['batters'] if isinstance(player_id, int))
                        # Add bench players
                        if 'bench' in team_data and isinstance(team_data['bench'], list):
                            all_scheduled_player_ids.update(player_id for player_id in team_data['bench'] if isinstance(player_id, int))
                        
                        # Populate player ID to name map from boxscore player details
                        if 'players' in team_data and isinstance(team_data['players'], dict):
                            for player_id_str, player_details in team_data['players'].items():
                                try:
                                    # Ensure we get the integer ID and full name from the 'person' object
                                    if 'person' in player_details and \
                                       isinstance(player_details['person'], dict) and \
                                       'id' in player_details['person'] and \
                                       'fullName' in player_details['person']:
                                        
                                        person_id = player_details['person']['id']
                                        full_name = player_details['person']['fullName']
                                        
                                        if isinstance(person_id, int) and isinstance(full_name, str):
                                            player_id_to_name_map[person_id] = full_name
                                            all_scheduled_player_ids.add(person_id) # Ensure it's added to global set too
                                            if team_type == 'home':
                                                current_game_home_player_ids.add(person_id)
                                            elif team_type == 'away':
                                                current_game_away_player_ids.add(person_id)
                                            # print(f"DEBUG: Added/Updated from boxscore: {person_id} - {full_name}") # Optional Debug
                                        # else:
                                            # print(f"DEBUG: person_id or fullName not expected type for {player_id_str}") # Optional Debug
                                    # else:
                                        # print(f"DEBUG: 'person' object or required keys missing for {player_id_str}") # Optional Debug
                                except Exception as e: # Broader catch for unexpected issues during parsing
                                    # print(f"Error parsing player details for {player_id_str} in game {game_pk}: {e}")
                                    pass
            else:
                print(f"Warning: Could not get detailed game info or valid teams data for roster extraction for game_pk: {game_pk}")
        # End of boxscore processing for this game for roster extraction

        # --- BEGIN DEBUG BLOCK for Boxscore --- 
        venue_data = game.get("venue", {})
        venue_id = venue_data.get("id")
        venue_name = venue_data.get("name", "Unknown Venue")
        
        # Store ballpark factor for this venue
        ballpark_factor = ballpark_hr_factors.get(venue_id, 1.0) if venue_id else 1.0
        
        home_team = game.get("teams", {}).get("home", {}).get("team", {}).get("name", "Home Team")
        away_team = game.get("teams", {}).get("away", {}).get("team", {}).get("name", "Away Team")
        
        print(f"Processing game {game_pk}: {home_team} vs {away_team}")
        
        # Check for pitcher data in the schedule response first
        home_team_data = game.get("teams", {}).get("home", {})
        away_team_data = game.get("teams", {}).get("away", {})
        
        home_pitcher_data = home_team_data.get("probablePitcher", {})
        if home_pitcher_data and home_pitcher_data.get('id') and home_pitcher_data.get('fullName'):
            player_id_to_name_map[home_pitcher_data['id']] = home_pitcher_data['fullName']
        away_pitcher_data = away_team_data.get("probablePitcher", {})
        if away_pitcher_data and away_pitcher_data.get('id') and away_pitcher_data.get('fullName'):
            player_id_to_name_map[away_pitcher_data['id']] = away_pitcher_data['fullName']
        
        # If probable pitchers not in schedule, fetch detailed game info
        if not home_pitcher_data or not away_pitcher_data:
            print(f"  Probable pitchers not found in schedule data, fetching detailed info...")
            detailed_game_info = get_game_detailed_info(game_pk)

            if detailed_game_info:
                # DEBUG: Print the structure of detailed_game_info (boxscore)
                print(f"DEBUG: Detailed Game Info (Boxscore) for game_pk: {game_pk}")
                print(json.dumps(detailed_game_info, indent=2))
                # For a cleaner exit, you could also return a jsonify message here.
                # return jsonify({"message": f"Debug: Printed boxscore for game {game_pk}, exiting."})
                # break # Exiting loop after first game
            
            if detailed_game_info:
                print(f"  Got detailed game info. Keys: {detailed_game_info.keys()}")
                
                # Extract pitcher information from detailed game data
                if 'teams' in detailed_game_info:
                    # Try to get starting pitchers from the detailed boxscore info
                    if 'home' in detailed_game_info['teams']:
                        home_pitchers = detailed_game_info['teams']['home'].get('pitchers', [])
                        if home_pitchers and len(home_pitchers) > 0:
                            # Use the first pitcher in the list as the starter
                            home_pitcher_id = home_pitchers[0]
                            home_pitcher_data = {'id': home_pitcher_id}
                            print(f"  Found home starting pitcher ID from detailed data: {home_pitcher_id}")
                    
                    if 'away' in detailed_game_info['teams']:
                        away_pitchers = detailed_game_info['teams']['away'].get('pitchers', [])
                        if away_pitchers and len(away_pitchers) > 0:
                            # Use the first pitcher in the list as the starter
                            away_pitcher_id = away_pitchers[0]
                            away_pitcher_data = {'id': away_pitcher_id}
                            print(f"  Found away starting pitcher ID from detailed data: {away_pitcher_id}")
        
        print(f"  Final home pitcher data: {home_pitcher_data}")
        print(f"  Final away pitcher data: {away_pitcher_data}")
        
        # Get weather data if coordinates are available
        weather_data = None
        if venue_data.get("location"):
            lat = venue_data.get("location", {}).get("latitude")
            lon = venue_data.get("location", {}).get("longitude")
            if lat and lon:
                try:
                    weather_data = get_weather_forecast(float(lat), float(lon))
                    games_weather_data[game_pk] = weather_data
                except (ValueError, TypeError) as e:
                    print(f"Error fetching weather data for venue {venue_name}: {e}")
        
        # Calculate weather impact
        weather_score = 0.0
        if weather_data:
            # Temperature impact (higher temp = more HRs)
            if 'hourly' in weather_data and 'temperature_2m' in weather_data['hourly'] and len(weather_data['hourly']['temperature_2m']) > 0:
                temp = weather_data['hourly']['temperature_2m'][0]
                weather_score += (temp - 70) * 0.01 if temp > 50 else -0.2
            
            # Wind impact (need to consider direction relative to ballpark orientation)
            if ('hourly' in weather_data and 'wind_speed_10m' in weather_data['hourly'] and 
                'wind_direction_10m' in weather_data['hourly'] and len(weather_data['hourly']['wind_speed_10m']) > 0):
                wind_speed = weather_data['hourly']['wind_speed_10m'][0]
                wind_direction = weather_data['hourly']['wind_direction_10m'][0]
                # Simplistic approach, should be customized per ballpark orientation
                # Increase score if wind direction is outward (180-360 degrees is often outward)
                if 180 <= wind_direction <= 360:
                    weather_score += wind_speed * 0.01
                else:
                    weather_score -= wind_speed * 0.01
        
            # Initialize sets for current game's players
            current_game_home_player_ids = set()
            current_game_away_player_ids = set()

            # Process players from detailed_game_info (boxscore)
            # This detailed_game_info should be the one fetched earlier in the loop for the current game_pk
            if detailed_game_info and isinstance(detailed_game_info, dict) and 'teams' in detailed_game_info:
                # Home team players for THIS specific game
                home_players_boxscore = detailed_game_info.get('teams', {}).get('home', {}).get('players', {})
                if isinstance(home_players_boxscore, dict):
                    for player_key, player_data_item in home_players_boxscore.items(): # Renamed to avoid conflict
                        if isinstance(player_data_item, dict) and 'person' in player_data_item and \
                           isinstance(player_data_item['person'], dict) and 'id' in player_data_item['person']:
                            person_id = player_data_item['person']['id']
                            if isinstance(person_id, int):
                                current_game_home_player_ids.add(person_id)
                                all_scheduled_player_ids.add(person_id) # Add to global set
                                if 'fullName' in player_data_item['person']:
                                    player_id_to_name_map[person_id] = player_data_item['person']['fullName']
                
                # Away team players for THIS specific game
                away_players_boxscore = detailed_game_info.get('teams', {}).get('away', {}).get('players', {})
                if isinstance(away_players_boxscore, dict):
                    for player_key, player_data_item in away_players_boxscore.items(): # Renamed to avoid conflict
                        if isinstance(player_data_item, dict) and 'person' in player_data_item and \
                           isinstance(player_data_item['person'], dict) and 'id' in player_data_item['person']:
                            person_id = player_data_item['person']['id']
                            if isinstance(person_id, int):
                                current_game_away_player_ids.add(person_id)
                                all_scheduled_player_ids.add(person_id) # Add to global set
                                if 'fullName' in player_data_item['person']:
                                    player_id_to_name_map[person_id] = player_data_item['person']['fullName']
            else:
                # This 'detailed_game_info' might be from the specific probable pitcher fallback logic
                # The main detailed_game_info for roster is typically fetched once per game if debug_first_game_boxscore_only is False
                # For robustness, we should ensure that the 'detailed_game_info' variable used here is the one intended for full roster.
                # This logic assumes 'detailed_game_info' variable at this point in the loop holds the boxscore for game_pk.
                print(f"  Warning: No comprehensive detailed_game_info or teams data for game {game_pk} at the point of player list extraction.")

            # Initialize sets for current game's players
            current_game_home_player_ids = set()
            current_game_away_player_ids = set()

            # Process players from detailed_game_info (boxscore)
            # This detailed_game_info should be the one fetched earlier in the loop for the current game_pk.
            # Ensure 'detailed_game_info' is the comprehensive boxscore at this stage.
            if detailed_game_info and isinstance(detailed_game_info, dict) and 'teams' in detailed_game_info:
                # Home team players for THIS specific game
                home_players_boxscore = detailed_game_info.get('teams', {}).get('home', {}).get('players', {})
                if isinstance(home_players_boxscore, dict):
                    for player_key, player_data_item in home_players_boxscore.items():
                        if isinstance(player_data_item, dict) and 'person' in player_data_item and \
                           isinstance(player_data_item['person'], dict) and 'id' in player_data_item['person']:
                            person_id = player_data_item['person']['id']
                            if isinstance(person_id, int):
                                current_game_home_player_ids.add(person_id)
                                all_scheduled_player_ids.add(person_id) # Add to global set
                                if 'fullName' in player_data_item['person']:
                                    player_id_to_name_map[person_id] = player_data_item['person']['fullName']
                
                # Away team players for THIS specific game
                away_players_boxscore = detailed_game_info.get('teams', {}).get('away', {}).get('players', {})
                if isinstance(away_players_boxscore, dict):
                    for player_key, player_data_item in away_players_boxscore.items():
                        if isinstance(player_data_item, dict) and 'person' in player_data_item and \
                           isinstance(player_data_item['person'], dict) and 'id' in player_data_item['person']:
                            person_id = player_data_item['person']['id']
                            if isinstance(person_id, int):
                                current_game_away_player_ids.add(person_id)
                                all_scheduled_player_ids.add(person_id) # Add to global set
                                if 'fullName' in player_data_item['person']:
                                    player_id_to_name_map[person_id] = player_data_item['person']['fullName']
            else:
                print(f"  Warning: No comprehensive detailed_game_info or teams data for game {game_pk} at the point of player list extraction. 'current_game_home/away_player_ids' will be empty for this game.")

            current_game_home_player_ids = set() # Ensure initialized
            current_game_away_player_ids = set() # Ensure initialized
        # Store all game info together
        game_data[game_pk] = {
            'home_team': home_team,
            'away_team': away_team,
            'venue': venue_name,
            'venue_id': venue_id,
            'ballpark_factor': ballpark_factor,
            'weather_score': weather_score,
            'home_pitcher': home_pitcher_data.get("id") if home_pitcher_data else None,
            'away_pitcher': away_pitcher_data.get("id") if away_pitcher_data else None,
            'home_team_players': current_game_home_player_ids,
            'away_team_players': current_game_away_player_ids
        }

    # Get all starting pitchers from the games
    scheduled_starting_pitcher_ids = set()
    print(f"Found {len(game_data)} games in schedule")
    
    # Debug: Print game data to see what we have
    for game_pk, game_info in game_data.items():
        print(f"Game {game_pk}: {game_info['home_team']} vs {game_info['away_team']} at {game_info['venue']}")
        print(f"  Home pitcher ID: {game_info['home_pitcher']}")
        print(f"  Away pitcher ID: {game_info['away_pitcher']}")
        
        if game_info['home_pitcher']:
            scheduled_starting_pitcher_ids.add(game_info['home_pitcher'])
        if game_info['away_pitcher']:
            scheduled_starting_pitcher_ids.add(game_info['away_pitcher'])
    
    print(f"Total pitcher IDs found: {len(scheduled_starting_pitcher_ids)}")
    if not scheduled_starting_pitcher_ids:
        print("WARNING: No pitcher IDs were found in the schedule data!")

    if not scheduled_starting_pitcher_ids:
        return jsonify({"message": f"No probable starting pitchers found for {date_str}."}), 200
    print(f"Scheduled starting pitcher IDs for {date_str}: {scheduled_starting_pitcher_ids}")

    print(f"Total unique players (batters/bench) found in boxscores for {date_str}: {len(all_scheduled_player_ids)}")

    # Get all unique batter IDs from the Statcast data
    all_batters_in_statcast_data = list(statcast_df_period['batter'].unique())
    print(f"Found {len(all_batters_in_statcast_data)} unique batters in Statcast period (pre-filtering).") # Debug

    active_batters_for_predictions = []
    if not all_scheduled_player_ids:
        print(f"Warning: No scheduled players found from boxscores for {date_str}. Predictions will be severely limited or impossible.")
        # Depending on desired behavior, could predict for all_batters_in_statcast_data or return error.
        # For now, we'll proceed, and if active_batters_for_predictions is empty, it will be caught later.
    else:
        # Ensure IDs are integers for comparison
        # Statcast 'batter' IDs are typically float64 from pybaseball, convert to int if possible
        # Boxscore IDs are integers
        all_batters_in_statcast_data_int = []
        for b_id in all_batters_in_statcast_data:
            try:
                all_batters_in_statcast_data_int.append(int(b_id))
            except (ValueError, TypeError):
                # print(f"Could not convert batter ID {b_id} to int from Statcast data.") # Optional debug
                pass # Skip if not convertible, or handle as error

        active_batters_for_predictions = [
            batter_id for batter_id in all_batters_in_statcast_data_int if batter_id in all_scheduled_player_ids
        ]
        print(f"Filtered to {len(active_batters_for_predictions)} batters who are both in Statcast data AND scheduled to play on {date_str}.")

    if not active_batters_for_predictions:
        print(f"No Statcast batters found who are also scheduled to play on {date_str}. No predictions can be made.")
        return jsonify({"error": f"No Statcast batters found among {len(all_scheduled_player_ids)} scheduled players for {date_str}. Ensure Statcast data covers these players and date range."}), 404

    predictions = []
    all_pitcher_overall_stats = {}
    print("Pre-calculating overall stats for scheduled pitchers...")
    for p_id in scheduled_starting_pitcher_ids:
        try:
            numeric_p_id = int(p_id)
            all_pitcher_overall_stats[numeric_p_id] = calculate_pitcher_overall_stats(statcast_df_period, numeric_p_id)
        except ValueError:
            print(f"Warning: Could not convert pitcher ID {p_id} to int. Skipping.")
            continue

    all_batter_overall_stats = {}
    print("Pre-calculating overall stats for all batters in Statcast period...")
    for b_id in all_batters_in_statcast_data:
        all_batter_overall_stats[b_id] = calculate_batter_overall_stats(statcast_df_period, b_id)

    print(f"Generating predictions for {len(active_batters_for_predictions)} batters against")

    # DEBUG: Inspect critical data before matchup loop
    print(f"DEBUG: Sample of all_batters_in_statcast_data (first 5 IDs): {list(all_batters_in_statcast_data)[:5] if all_batters_in_statcast_data else 'Empty'}")
    if game_data:
        first_game_pk_for_debug = list(game_data.keys())[0]
        # print(f"DEBUG: Full game_data entry for first game_pk {first_game_pk_for_debug}: {game_data[first_game_pk_for_debug]}") # Can be very verbose
        print(f"DEBUG: Keys in game_data for first game_pk {first_game_pk_for_debug}: {game_data[first_game_pk_for_debug].keys()}")
        print(f"DEBUG: Home players in game_data for game {first_game_pk_for_debug} (first 5): {list(game_data[first_game_pk_for_debug].get('home_team_players', set()))[:5]}")
        print(f"DEBUG: Away players in game_data for game {first_game_pk_for_debug} (first 5): {list(game_data[first_game_pk_for_debug].get('away_team_players', set()))[:5]}")
    else:
        print("DEBUG: game_data is empty before matchup loop!")

    # Generate matchups for games
    all_matchups = []
    first_iteration_debug_done = False # Flag to print details only for the first game processed
    for game_pk_iter, gd_item in game_data.items(): # Iterate through each game
        home_pitcher_id = gd_item.get('home_pitcher') # Corrected: No '_id' suffix to match game_data population
        away_pitcher_id = gd_item.get('away_pitcher') # Corrected: No '_id' suffix to match game_data population
        
        home_team_players_in_game = gd_item.get('home_team_players', set())
        away_team_players_in_game = gd_item.get('away_team_players', set())

        if not first_iteration_debug_done:
            print(f"--- DEBUG: First Iteration of Matchup Loop for game_pk: {game_pk_iter} ---")
            print(f"  Home Pitcher ID: {home_pitcher_id}, Away Pitcher ID: {away_pitcher_id}")
            print(f"  Home Team Players in this game (RAW from gd_item, first 10 of {len(home_team_players_in_game)}): {list(home_team_players_in_game)[:10] if home_team_players_in_game else 'Empty'}")
            print(f"  Away Team Players in this game (RAW from gd_item, first 10 of {len(away_team_players_in_game)}): {list(away_team_players_in_game)[:10] if away_team_players_in_game else 'Empty'}")
            
            intersect_home_with_statcast = home_team_players_in_game.intersection(all_batters_in_statcast_data)
            print(f"  Intersection (Home Players in Game & All Statcast Batters) ({len(intersect_home_with_statcast)}): {list(intersect_home_with_statcast)[:10] if intersect_home_with_statcast else 'Empty'}")
            
            intersect_away_with_statcast = away_team_players_in_game.intersection(all_batters_in_statcast_data)
            print(f"  Intersection (Away Players in Game & All Statcast Batters) ({len(intersect_away_with_statcast)}): {list(intersect_away_with_statcast)[:10] if intersect_away_with_statcast else 'Empty'}")
            print(f"--- End First Iteration Debug ---")
            first_iteration_debug_done = True

        # Match home team batters against the away team's pitcher for THIS game
        if away_pitcher_id:
            for batter_id in home_team_players_in_game:
                if batter_id in all_batters_in_statcast_data: # Ensure we have historical data for this batter
                    all_matchups.append((batter_id, away_pitcher_id, game_pk_iter))
                    # print(f"DEBUG: Matchup: Home Batter {batter_id} vs Away Pitcher {away_pitcher_id} in game {game_pk_iter}")

        # Match away team batters against the home team's pitcher for THIS game
        if home_pitcher_id:
            for batter_id in away_team_players_in_game:
                if batter_id in all_batters_in_statcast_data: # Ensure we have historical data for this batter
                    all_matchups.append((batter_id, home_pitcher_id, game_pk_iter))
                    # print(f"DEBUG: Matchup: Away Batter {batter_id} vs Home Pitcher {home_pitcher_id} in game {game_pk_iter}")
    
    # Calculate handedness splits
    all_batter_handedness_stats = {}
    print("Calculating batter handedness stats...")
    for batter_id in all_batters_in_statcast_data:
        # Calculate stats for both RHP and LHP
        rhp_stats = calculate_batter_handedness_stats(statcast_df_period, batter_id, pitcher_throws='R')
        lhp_stats = calculate_batter_handedness_stats(statcast_df_period, batter_id, pitcher_throws='L')
        all_batter_handedness_stats[batter_id] = {
            'vs_rhp': rhp_stats if not rhp_stats.get('error') else None,
            'vs_lhp': lhp_stats if not lhp_stats.get('error') else None
        }
        
    all_pitcher_handedness_stats = {}
    print("Calculating pitcher handedness stats...")
    for pitcher_id in scheduled_starting_pitcher_ids:
        try:
            numeric_pitcher_id = int(pitcher_id)
            # Calculate stats for both RHB and LHB
            rhb_stats = calculate_pitcher_handedness_stats(statcast_df_period, numeric_pitcher_id, batter_stands='R')
            lhb_stats = calculate_pitcher_handedness_stats(statcast_df_period, numeric_pitcher_id, batter_stands='L')
            all_pitcher_handedness_stats[numeric_pitcher_id] = {
                'vs_rhb': rhb_stats if not rhb_stats.get('error') else None,
                'vs_lhb': lhb_stats if not lhb_stats.get('error') else None
            }
        except ValueError:
            print(f"Warning: Could not convert pitcher ID {pitcher_id} to int for handedness stats. Skipping.")
            continue

    # Generate predictions for each matchup
    predictions = []
    # Deduplicate matchups, as a safeguard (though the new logic should prevent most duplicates)
    unique_matchups = list(set(all_matchups))
    print(f"DEBUG: Total raw matchups generated: {len(all_matchups)}, Unique matchups: {len(unique_matchups)}")
    all_matchups = unique_matchups
    
    print(f"Generated {len(all_matchups)} total potential matchups for BvP calculation.")

    if not all_matchups:
        return jsonify({"error": f"No matchups could be generated for {date_str}. This might be due to no overlapping players between Statcast data and game schedules, or no probable pitchers listed."}), 404
    for batter_id, pitcher_id, game_pk in all_matchups:
        try:
            # Convert IDs to integers for dictionary access
            numeric_batter_id = int(batter_id)
            numeric_pitcher_id = int(pitcher_id)
            
            # Get basic metrics
            batter_metrics = all_batter_overall_stats.get(numeric_batter_id)
            pitcher_metrics = all_pitcher_overall_stats.get(numeric_pitcher_id)
            
            if not batter_metrics or 'error' in batter_metrics or not pitcher_metrics or 'error' in pitcher_metrics:
                continue
            
            # Get handedness data
            batter_handedness = all_batter_handedness_stats.get(numeric_batter_id, {})
            pitcher_handedness = all_pitcher_handedness_stats.get(numeric_pitcher_id, {})
            
            # Get game-specific environmental factors
            game_info = game_data.get(game_pk, {})
            game_venue_name = game_info.get('venue', 'Unknown Venue') 
            ballpark_factor = ballpark_hr_factors.get(game_venue_name, DEFAULT_BALLPARK_FACTOR)
            weather_score = games_weather_data.get(game_pk, {}).get('weather_impact_score', 0) # Corrected to use games_weather_data with game_pk
            print(f"  [Game {game_pk}] Venue: '{game_venue_name}', Ballpark Factor: {ballpark_factor}, Weather Score: {weather_score}") # Log for verification
            
            # Determine handedness matchup statistics to use
            pitcher_throws = pitcher_handedness.get('pitcher_throws')
            batter_stands = batter_handedness.get('batter_stands')
            
            # Default to overall metrics
            handedness_adjusted_batter_metrics = batter_metrics.copy() if batter_metrics else {}
            handedness_adjusted_pitcher_metrics = pitcher_metrics.copy() if pitcher_metrics else {}
            
            # Calculate handedness-specific stats for batter against this pitcher handedness
            if pitcher_throws in ['R', 'L'] and numeric_batter_id:
                batter_handedness_stats = calculate_batter_handedness_stats(
                    statcast_df_period, 
                    numeric_batter_id, 
                    pitcher_throws=pitcher_throws
                )
                # Only update if we have valid stats (no error)
                if batter_handedness_stats and not batter_handedness_stats.get('error'):
                    # Add these stats to the batter metrics
                    handedness_adjusted_batter_metrics.update({
                        'home_run_rate_vs_handedness': batter_handedness_stats.get('home_run_rate_vs_handedness', 0),
                        'iso_vs_handedness': batter_handedness_stats.get('iso_vs_handedness', 0),
                        'barrel_pct_vs_handedness': batter_handedness_stats.get('barrel_pct_vs_handedness', 0)
                    })
                    print(f"Added handedness stats for batter {numeric_batter_id} vs {pitcher_throws}-handed pitchers")
            
            # Calculate handedness-specific stats for pitcher against this batter handedness
            if batter_stands in ['R', 'L'] and numeric_pitcher_id:
                pitcher_handedness_stats = calculate_pitcher_handedness_stats(
                    statcast_df_period, 
                    numeric_pitcher_id, 
                    batter_stands=batter_stands
                )
                # Only update if we have valid stats (no error)
                if pitcher_handedness_stats and not pitcher_handedness_stats.get('error'):
                    # Add these stats to the pitcher metrics
                    handedness_adjusted_pitcher_metrics.update({
                        'home_run_rate_allowed_vs_handedness': pitcher_handedness_stats.get('home_run_rate_allowed_vs_handedness', 0),
                        'barrel_pct_allowed_vs_handedness': pitcher_handedness_stats.get('barrel_pct_allowed_vs_handedness', 0)
                    })
                    print(f"Added handedness stats for pitcher {numeric_pitcher_id} vs {batter_stands}-handed batters")
            

            # Add batter handedness ('stand')
            if numeric_batter_id and 'batter' in statcast_df_period.columns:
                batter_specific_rows = statcast_df_period[statcast_df_period['batter'] == numeric_batter_id]
                if not batter_specific_rows.empty and 'stand' in batter_specific_rows.columns:
                    batter_stands_series = batter_specific_rows['stand'].dropna()
                    if not batter_stands_series.empty:
                        handedness_adjusted_batter_metrics['batter_stands'] = batter_stands_series.iloc[0]
                    else:
                        handedness_adjusted_batter_metrics['batter_stands'] = 'N/A'
                else:
                    handedness_adjusted_batter_metrics['batter_stands'] = 'N/A'
            else:
                handedness_adjusted_batter_metrics['batter_stands'] = 'N/A'

            # Add pitcher handedness ('p_throws')
            if numeric_pitcher_id and 'pitcher' in statcast_df_period.columns:
                pitcher_specific_rows = statcast_df_period[statcast_df_period['pitcher'] == numeric_pitcher_id]
                if not pitcher_specific_rows.empty and 'p_throws' in pitcher_specific_rows.columns:
                    pitcher_throws_series = pitcher_specific_rows['p_throws'].dropna()
                    if not pitcher_throws_series.empty:
                        handedness_adjusted_pitcher_metrics['pitcher_throws'] = pitcher_throws_series.iloc[0]
                    else:
                        handedness_adjusted_pitcher_metrics['pitcher_throws'] = 'N/A'
                else:
                    handedness_adjusted_pitcher_metrics['pitcher_throws'] = 'N/A'
            else:
                handedness_adjusted_pitcher_metrics['pitcher_throws'] = 'N/A'

            # Calculate the HR likelihood score
            score = generate_hr_likelihood_score(
                batter_metrics=handedness_adjusted_batter_metrics,
                pitcher_metrics=handedness_adjusted_pitcher_metrics,
                ballpark_factor=ballpark_factor,
                weather_score=weather_score
            )
            
            # Append prediction with all data
            prediction_data = {
                'game_pk': game_pk,
                'batter_id': make_json_serializable(numeric_batter_id),
                'pitcher_id': make_json_serializable(numeric_pitcher_id),
                'batter_name': player_id_to_name_map.get(numeric_batter_id, f"Player {numeric_batter_id}"),
                'pitcher_name': player_id_to_name_map.get(numeric_pitcher_id, f"Player {numeric_pitcher_id}"),
                'bat_hand': batter_stands,  # Renamed from batter_stands
                'pitch_hand': pitcher_throws,  # Renamed from pitcher_throws
                'venue_name': game_info.get('venue', 'Unknown Venue'),  # Renamed from venue
                'home_team': game_info.get('home_team', 'Home Team'),
                'away_team': game_info.get('away_team', 'Away Team'),
                'hr_likelihood_score': make_json_serializable(score),
                'ballpark_factor': make_json_serializable(ballpark_factor),
                'weather_score': make_json_serializable(weather_score),
                'batter_overall_stats': make_json_serializable(batter_metrics),
                'handedness_adjusted_batter_stats': make_json_serializable(handedness_adjusted_batter_metrics),
                'pitcher_overall_stats': make_json_serializable(pitcher_metrics),
                'handedness_adjusted_pitcher_stats': make_json_serializable(handedness_adjusted_pitcher_metrics)
            }
            predictions.append(prediction_data)

        except Exception as e:
            print(f"Error processing matchup {batter_id} vs {pitcher_id}: {e}")
            continue
            
    # Sort predictions by likelihood score
    predictions.sort(key=lambda x: x['hr_likelihood_score'], reverse=True)
    print(f"Generated {len(predictions)} predictions. Top score: {predictions[0]['hr_likelihood_score'] if predictions else 'N/A'}")
    
    # Limit results if requested
    if limit and isinstance(limit, int) and limit > 0:
        limited_predictions = predictions[:limit]
        print(f"Returning top {limit} predictions out of {len(predictions)} total")
        
        # Restore original stdout
        sys.stdout = old_stdout
        
        # Get the captured logs
        logs = log_capture.getvalue()
        # No need to print a sample since we're already printing to console
        
        # Include logs in the response
        response = {
            'predictions': limited_predictions,
            'logs': logs
        }
        return jsonify(response)
        
    # Restore original stdout
    sys.stdout = old_stdout
    
    # Get the captured logs
    logs = log_capture.getvalue()
    # No need to print a sample since we're already printing to console
    
    # Include logs in the response
    response = {
        'predictions': predictions,
        'logs': logs
    }
    return jsonify(response)

# Helper function to convert NumPy types to Python native types for JSON serialization
def convert_numpy_types(obj):
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(i) for i in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

@app.route('/api/debug/player-stats', methods=['GET'])
def debug_player_stats():
    # Get query parameters
    player_id = request.args.get('player_id')
    player_type = request.args.get('player_type', 'batter')  # Default to batter
    days_back = int(request.args.get('days_back', 365))  # Default to 365 days
    
    if not player_id:
        return jsonify({'error': 'Player ID is required'}), 400
        
    if player_type not in ['batter', 'pitcher']:
        return jsonify({'error': 'Player type must be either "batter" or "pitcher"'}), 400
    
    try:
        # Convert to numeric ID if needed
        numeric_player_id = int(player_id)
        
        # Calculate date range
        end_date = datetime.now().strftime('%Y-%m-%d')
        # make start date march 1st 2025
        start_date = '2025-03-01'
        
        print(f"Fetching Statcast data for {player_type} {numeric_player_id} from {start_date} to {end_date}")
        
        # Get Statcast data
        statcast_df = get_statcast_data_for_date_range(start_date, end_date)
        
        if statcast_df.empty:
            return jsonify({'error': 'No Statcast data available for the specified date range'}), 404
        
        # Initialize variables
        rhp_stats = {}
        lhp_stats = {}
        rhb_stats = {}
        lhb_stats = {}
        overall_stats = {}
        
        if player_type == 'batter':
            # Calculate batter stats vs both handedness
            rhp_stats = calculate_batter_handedness_stats(statcast_df, numeric_player_id, pitcher_throws='R')
            lhp_stats = calculate_batter_handedness_stats(statcast_df, numeric_player_id, pitcher_throws='L')
            
            # Get overall batter stats for comparison
            overall_stats = calculate_batter_overall_stats(statcast_df, numeric_player_id)
        else:  # pitcher
            # Calculate pitcher stats vs both handedness
            rhb_stats = calculate_pitcher_handedness_stats(statcast_df, numeric_player_id, batter_stands='R')
            lhb_stats = calculate_pitcher_handedness_stats(statcast_df, numeric_player_id, batter_stands='L')
            
            # Get overall pitcher stats for comparison
            overall_stats = calculate_pitcher_overall_stats(statcast_df, numeric_player_id)
        
        # Get player info for additional context
        # We need to use the MLB API to get player info by ID
        # Since find_player_ids only works with names, we'll use a different approach
        try:
            endpoint = f"https://statsapi.mlb.com/api/v1/people/{numeric_batter_id}"
            response = requests.get(endpoint)
            response.raise_for_status()
            player_data = response.json()
            player_info = player_data.get('people', [])
        except Exception as e:
            print(f"Error getting player info for ID {numeric_batter_id}: {e}")
            player_info = []
        
        # Format the response based on player type
        if player_type == 'batter':
            response = {
                'player_type': 'batter',
                'player_id': numeric_player_id,
                'player_name': player_info[0].get('fullName', 'Unknown') if player_info else 'Unknown',
                'batter_stands': player_info[0].get('batSide', {}).get('code', 'Unknown') if player_info else 'Unknown',
                'days_of_data': days_back,
                'overall_stats': overall_stats,
                'vs_rhp_stats': rhp_stats,
                'vs_lhp_stats': lhp_stats,
                'has_rhp_error': 'error' in rhp_stats,
                'has_lhp_error': 'error' in lhp_stats,
                'rhp_error_message': rhp_stats.get('error', None),
                'lhp_error_message': lhp_stats.get('error', None),
                'pa_vs_rhp': rhp_stats.get('pa_count', 0),
                'pa_vs_lhp': lhp_stats.get('pa_count', 0)
            }
        else:  # pitcher
            response = {
                'player_type': 'pitcher',
                'player_id': numeric_player_id,
                'player_name': player_info[0].get('fullName', 'Unknown') if player_info else 'Unknown',
                'pitcher_throws': player_info[0].get('pitchHand', {}).get('code', 'Unknown') if player_info else 'Unknown',
                'days_of_data': days_back,
                'overall_stats': overall_stats,
                'vs_rhb_stats': rhb_stats,
                'vs_lhb_stats': lhb_stats,
                'has_rhb_error': 'error' in rhb_stats,
                'has_lhb_error': 'error' in lhb_stats,
                'rhb_error_message': rhb_stats.get('error', None),
                'lhb_error_message': lhb_stats.get('error', None),
                'bf_vs_rhb': rhb_stats.get('bf_count', 0),
                'bf_vs_lhb': lhb_stats.get('bf_count', 0)
            }
        
        # Print detailed debug info to server logs
        if player_type == 'batter':
            print(f"DEBUG - Batter Stats for {response['player_name']} (ID: {numeric_player_id}):")
            print(f"Bats: {response['batter_stands']}")
            print(f"PAs vs RHP: {response['pa_vs_rhp']}")
            print(f"PAs vs LHP: {response['pa_vs_lhp']}")
            print(f"RHP Error: {response['rhp_error_message']}")
            print(f"LHP Error: {response['lhp_error_message']}")
            
            # Print key batter stats comparison
            if not response['has_rhp_error']:
                print("VS RHP Stats:")
                print(f"HR Rate: {rhp_stats.get('home_run_rate_vs_handedness', 0):.3f}")
                print(f"ISO: {rhp_stats.get('iso_vs_handedness', 0):.3f}")
                print(f"Barrel %: {rhp_stats.get('barrel_pct_vs_handedness', 0):.3f}")
            
            if not response['has_lhp_error']:
                print("VS LHP Stats:")
                print(f"HR Rate: {lhp_stats.get('home_run_rate_vs_handedness', 0):.3f}")
                print(f"ISO: {lhp_stats.get('iso_vs_handedness', 0):.3f}")
                print(f"Barrel %: {lhp_stats.get('barrel_pct_vs_handedness', 0):.3f}")
                
            print("Overall Batter Stats:")
            print(f"HR Rate: {overall_stats.get('home_run_rate_per_pa', 0):.3f}")
            print(f"ISO: {overall_stats.get('iso', 0):.3f}")
            print(f"Barrel %: {overall_stats.get('barrel_pct', 0):.3f}")
        else:  # pitcher
            print(f"DEBUG - Pitcher Stats for {response['player_name']} (ID: {numeric_player_id}):")
            print(f"Throws: {response['pitcher_throws']}")
            print(f"BF vs RHB: {response['bf_vs_rhb']}")
            print(f"BF vs LHB: {response['bf_vs_lhb']}")
            print(f"RHB Error: {response['rhb_error_message']}")
            print(f"LHB Error: {response['lhb_error_message']}")
            
            # Print key pitcher stats comparison
            if not response['has_rhb_error']:
                print("VS RHB Stats:")
                print(f"HR Rate Allowed: {rhb_stats.get('home_run_rate_allowed_vs_handedness', 0):.3f}")
                print(f"Barrel % Allowed: {rhb_stats.get('barrel_pct_allowed_vs_handedness', 0):.3f}")
            
            if not response['has_lhb_error']:
                print("VS LHB Stats:")
                print(f"HR Rate Allowed: {lhb_stats.get('home_run_rate_allowed_vs_handedness', 0):.3f}")
                print(f"Barrel % Allowed: {lhb_stats.get('barrel_pct_allowed_vs_handedness', 0):.3f}")
                
            print("Overall Pitcher Stats:")
            if 'home_run_rate_allowed' in overall_stats:
                print(f"HR Rate Allowed: {overall_stats.get('home_run_rate_allowed', 0):.3f}")
            if 'barrel_pct_allowed' in overall_stats:
                print(f"Barrel % Allowed: {overall_stats.get('barrel_pct_allowed', 0):.3f}")
        
        # Convert NumPy types to Python native types for JSON serialization
        serializable_response = convert_numpy_types(response)
        return jsonify(serializable_response)
        
    except Exception as e:
        print(f"Error in debug_batter_handedness_stats: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
