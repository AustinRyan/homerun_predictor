import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score, brier_score_loss
import itertools
import logging

# Import local modules
from algorithm.predictor import (
    generate_hr_likelihood_score,
    calculate_batter_overall_stats,
    calculate_pitcher_overall_stats
)
from data_fetchers.mlb_api import get_mlb_schedule, get_statcast_data_for_date_range

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def calculate_performance_metrics(predicted_scores, actual_outcomes):
    """
    Calculate various performance metrics for binary classification.
    
    Args:
        predicted_scores: List of predicted HR likelihood scores (0-100)
        actual_outcomes: List of actual outcomes (1 for HR, 0 for no HR)
    
    Returns:
        Dictionary of metrics
    """
    # Normalize scores to 0-1 range for probability metrics
    normalized_scores = [score/100.0 for score in predicted_scores]
    
    metrics = {}
    
    # ROC-AUC
    try:
        metrics['roc_auc'] = roc_auc_score(actual_outcomes, normalized_scores)
    except Exception as e:
        logger.warning(f"Could not calculate ROC-AUC: {e}")
        metrics['roc_auc'] = None
    
    # Precision-Recall AUC
    try:
        metrics['pr_auc'] = average_precision_score(actual_outcomes, normalized_scores)
    except Exception as e:
        logger.warning(f"Could not calculate PR-AUC: {e}")
        metrics['pr_auc'] = None
    
    # Calibration (Brier score - lower is better)
    try:
        metrics['brier_score'] = brier_score_loss(actual_outcomes, normalized_scores)
    except Exception as e:
        logger.warning(f"Could not calculate Brier score: {e}")
        metrics['brier_score'] = None
    
    # Compute precision, recall at different thresholds
    try:
        precision, recall, thresholds = precision_recall_curve(actual_outcomes, normalized_scores)
        
        # Find threshold that maximizes F1 score
        f1_scores = [2 * (p * r) / (p + r) if (p + r) > 0 else 0 for p, r in zip(precision, recall)]
        best_f1_idx = np.argmax(f1_scores)
        
        metrics['best_threshold'] = thresholds[best_f1_idx] if best_f1_idx < len(thresholds) else 0.5
        metrics['best_f1'] = f1_scores[best_f1_idx]
        metrics['best_precision'] = precision[best_f1_idx]
        metrics['best_recall'] = recall[best_f1_idx]
    except Exception as e:
        logger.warning(f"Could not calculate precision-recall metrics: {e}")
        metrics['best_threshold'] = 0.5
        metrics['best_f1'] = None
        metrics['best_precision'] = None
        metrics['best_recall'] = None
    
    return metrics


def calculate_score_with_weights(batter_metrics, pitcher_metrics, custom_weights, ballpark_factor=1.0, weather_score=0):
    """
    Calculate HR likelihood score using custom weights.
    This is a simplified version of generate_hr_likelihood_score that accepts custom weights.
    
    Args:
        batter_metrics (dict): Batter statistics
        pitcher_metrics (dict): Pitcher statistics
        custom_weights (dict): Custom weights to use for calculation
        ballpark_factor (float): Ballpark factor for HRs
        weather_score (float): Weather impact score
        
    Returns:
        float: HR likelihood score (0-100)
    """
    # Use the existing function but override the weights
    # We need to modify the predictor.py function to accept custom weights
    # For now, we'll call the original function
    return generate_hr_likelihood_score(
        batter_metrics=batter_metrics,
        pitcher_metrics=pitcher_metrics,
        ballpark_factor=ballpark_factor,
        weather_score=weather_score,
        custom_weights=custom_weights  # This parameter needs to be added to generate_hr_likelihood_score
    )


def optimize_weights(historical_matchups_df, metric_to_optimize='roc_auc'):
    """
    Optimize weights based on historical performance.
    
    Args:
        historical_matchups_df: DataFrame with historical matchups data.
                               Must contain batter_metrics, pitcher_metrics, and hit_hr columns.
        metric_to_optimize: Metric to optimize ('roc_auc', 'pr_auc', or 'brier_score')
        
    Returns:
        tuple: (best_weights, best_metric_value)
    """
    logger.info(f"Starting weight optimization for {len(historical_matchups_df)} historical matchups")
    
    # Define the metrics we want to optimize and their weights
    metric_names = [
        'batter_avg_launch_speed', 'batter_ideal_launch_angle_rate', 'batter_hard_hit_rate',
        'batter_home_run_rate_per_pa', 'batter_iso', 'batter_barrel_pct', 'batter_sweet_spot_pct',
        'batter_exit_velo_consistency', 'pitcher_avg_launch_speed_allowed',
        'pitcher_ideal_launch_angle_rate_allowed', 'pitcher_hard_hit_rate_allowed',
        'pitcher_home_run_rate_allowed_per_pa', 'pitcher_fip', 'pitcher_barrel_pct_allowed',
        'pitcher_sweet_spot_pct_allowed', 'pitcher_fly_ball_rate_allowed',
        'pitcher_ground_ball_rate_allowed', 'handedness_advantage_bonus'
    ]
    
    # Define possible weight values to try for each metric
    # For efficiency, we'll use a smaller set of values for initial optimization
    # This can be expanded for more granular optimization
    possible_weights = {}
    
    # Key metrics with wider range
    for metric in ['batter_home_run_rate_per_pa', 'batter_barrel_pct', 'pitcher_barrel_pct_allowed',
                  'pitcher_home_run_rate_allowed_per_pa', 'batter_iso', 'pitcher_fip']:
        possible_weights[metric] = [1.0, 3.0, 6.0, 10.0]
    
    # Secondary metrics with narrower range
    for metric in ['batter_ideal_launch_angle_rate', 'batter_hard_hit_rate',
                  'pitcher_ideal_launch_angle_rate_allowed', 'pitcher_hard_hit_rate_allowed',
                  'batter_sweet_spot_pct', 'pitcher_sweet_spot_pct_allowed']:
        possible_weights[metric] = [0.5, 1.0, 1.5, 2.0]
    
    # Tertiary metrics with small range
    for metric in ['batter_avg_launch_speed', 'pitcher_avg_launch_speed_allowed',
                  'pitcher_fly_ball_rate_allowed', 'pitcher_ground_ball_rate_allowed',
                  'handedness_advantage_bonus', 'batter_exit_velo_consistency']:
        possible_weights[metric] = [-0.5, -0.2, 0.0, 0.1, 0.2, 0.5]
    
    best_metric_value = -float('inf') if metric_to_optimize != 'brier_score' else float('inf')
    best_weights = {}
    
    # Instead of trying all combinations (which would be too many),
    # we'll use a greedy approach where we optimize one weight at a time
    # Start with default weights
    current_weights = {
        'batter_avg_launch_speed': 0.1,
        'batter_ideal_launch_angle_rate': 1.5,
        'batter_hard_hit_rate': 1.5,
        'batter_home_run_rate_per_pa': 10.0,
        'batter_iso': 3.0,
        'batter_barrel_pct': 6.0,
        'batter_sweet_spot_pct': 1.0,
        'batter_exit_velo_consistency': -0.2,
        'pitcher_avg_launch_speed_allowed': 0.1,
        'pitcher_ideal_launch_angle_rate_allowed': 1.5,
        'pitcher_hard_hit_rate_allowed': 1.5,
        'pitcher_home_run_rate_allowed_per_pa': 10.0,
        'pitcher_fip': 3.0,
        'pitcher_barrel_pct_allowed': 6.0,
        'pitcher_sweet_spot_pct_allowed': 1.0,
        'pitcher_fly_ball_rate_allowed': 0.1,
        'pitcher_ground_ball_rate_allowed': -0.1,
        'handedness_advantage_bonus': 0.15
    }
    
    # Iterate through each metric and find the best weight for it
    for metric in metric_names:
        logger.info(f"Optimizing weight for {metric}")
        best_weight_for_metric = current_weights[metric]
        
        for weight in possible_weights[metric]:
            # Create a copy of current weights and update the current metric's weight
            test_weights = current_weights.copy()
            test_weights[metric] = weight
            
            # Calculate scores for all matchups with these weights
            scores = []
            for _, matchup in historical_matchups_df.iterrows():
                score = calculate_score_with_weights(
                    matchup['batter_metrics'],
                    matchup['pitcher_metrics'],
                    test_weights,
                    matchup.get('ballpark_factor', 1.0),
                    matchup.get('weather_score', 0)
                )
                scores.append(score)
            
            # Evaluate performance
            metrics = calculate_performance_metrics(scores, historical_matchups_df['hit_hr'])
            current_metric_value = metrics.get(metric_to_optimize)
            
            if current_metric_value is None:
                continue
                
            # Check if this is better than our current best
            # For brier_score, lower is better; for others, higher is better
            is_better = False
            if metric_to_optimize == 'brier_score':
                is_better = current_metric_value < best_metric_value
            else:
                is_better = current_metric_value > best_metric_value
                
            if is_better:
                best_metric_value = current_metric_value
                best_weight_for_metric = weight
                logger.info(f"  New best {metric_to_optimize}: {best_metric_value} with {metric}={weight}")
        
        # Update the current weights with the best weight for this metric
        current_weights[metric] = best_weight_for_metric
    
    # Final evaluation with the best weights
    scores = []
    for _, matchup in historical_matchups_df.iterrows():
        score = calculate_score_with_weights(
            matchup['batter_metrics'],
            matchup['pitcher_metrics'],
            current_weights,
            matchup.get('ballpark_factor', 1.0),
            matchup.get('weather_score', 0)
        )
        scores.append(score)
        
    final_metrics = calculate_performance_metrics(scores, historical_matchups_df['hit_hr'])
    logger.info(f"Optimization complete. Final {metric_to_optimize}: {final_metrics.get(metric_to_optimize)}")
    logger.info(f"Best weights: {current_weights}")
    
    return current_weights, final_metrics


def get_historical_matchups(start_date, end_date):
    """
    Retrieve historical matchups for a date range and prepare them for backtesting.
    
    Args:
        start_date (datetime): Start date for historical data
        end_date (datetime): End date for historical data
        
    Returns:
        pd.DataFrame: DataFrame with historical matchups and outcomes
    """
    logger.info(f"Retrieving historical matchups from {start_date} to {end_date}")
    
    # Format dates for API calls
    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')
    
    # Get Statcast data for the period (for outcomes)
    statcast_df_outcomes = get_statcast_data_for_date_range(start_date_str, end_date_str)
    if statcast_df_outcomes is None or statcast_df_outcomes.empty:
        logger.error(f"Could not fetch Statcast data for outcomes period {start_date_str} to {end_date_str}")
        return pd.DataFrame()
    
    # Get historical Statcast data for calculating player metrics (up to the day before start_date)
    historical_end_date = start_date - timedelta(days=1)
    historical_start_date = historical_end_date - timedelta(days=365)  # Use last year's data
    historical_start_str = historical_start_date.strftime('%Y-%m-%d')
    historical_end_str = historical_end_date.strftime('%Y-%m-%d')
    
    statcast_df_historical = get_statcast_data_for_date_range(historical_start_str, historical_end_str)
    if statcast_df_historical is None or statcast_df_historical.empty:
        logger.error(f"Could not fetch historical Statcast data for period {historical_start_str} to {historical_end_str}")
        return pd.DataFrame()
    
    # Define ballpark factors dictionary (same as in app.py)
    ballpark_hr_factors = {
        # American League
        "Yankee Stadium": 1.15,
        "Fenway Park": 0.90,
        "Oriole Park at Camden Yards": 0.85,
        "Tropicana Field": 0.95,
        "Rogers Centre": 1.00,
        "Guaranteed Rate Field": 1.05,
        "Progressive Field": 0.85,
        "Comerica Park": 0.90,
        "Kauffman Stadium": 0.95,
        "Target Field": 1.05,
        "Daikin Park": 1.10,
        "Sutter Health Park": 1.45,
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
        "loanDepot park": 0.90,
        "Truist Park": 1.00,
        "Estadio Alfredo Harp Helú": 1.20,
        "London Stadium": 1.10,
    }
    
    # Default factor for parks not listed
    DEFAULT_BALLPARK_FACTOR = 1.0
    
    # Process each day in the range
    all_matchups = []
    current_date = start_date
    while current_date <= end_date:
        current_date_str = current_date.strftime('%Y-%m-%d')
        logger.info(f"Processing matchups for {current_date_str}")
        
        # Get schedule for the day
        schedule_data = get_mlb_schedule(current_date_str)
        if not schedule_data or not schedule_data.get("dates") or not schedule_data["dates"][0].get("games"):
            logger.warning(f"No games found for {current_date_str}")
            current_date += timedelta(days=1)
            continue
        
        # Filter Statcast data for the current date to find outcomes
        current_date_statcast = statcast_df_outcomes[statcast_df_outcomes['game_date'] == current_date_str]
        
        # Process each game
        for game in schedule_data["dates"][0].get("games", []):
            game_pk = game.get("gamePk")
            if not game_pk:
                continue
            
            # Extract venue information
            venue_data = game.get("venue", {})
            venue_name = venue_data.get("name", "Unknown Venue")
            
            # Get ballpark factor
            ballpark_factor = ballpark_hr_factors.get(venue_name, DEFAULT_BALLPARK_FACTOR)
            
            # Extract home and away team info
            home_team_data = game.get("teams", {}).get("home", {})
            away_team_data = game.get("teams", {}).get("away", {})
            
            # Get probable pitchers
            home_pitcher_id = home_team_data.get("probablePitcher", {}).get("id")
            away_pitcher_id = away_team_data.get("probablePitcher", {}).get("id")
            
            if not home_pitcher_id or not away_pitcher_id:
                logger.warning(f"Missing probable pitcher for game {game_pk}")
                continue
            
            # Calculate pitcher stats using historical data
            home_pitcher_stats = calculate_pitcher_overall_stats(statcast_df_historical, home_pitcher_id)
            away_pitcher_stats = calculate_pitcher_overall_stats(statcast_df_historical, away_pitcher_id)
            
            # Filter Statcast data for this specific game
            game_statcast = current_date_statcast[current_date_statcast['game_pk'] == game_pk]
            
            # Get all batters who faced the away pitcher (home team batters)
            home_batters = game_statcast[(game_statcast['pitcher'] == away_pitcher_id) & 
                                       (game_statcast['inning_topbot'] == 'Bot')]['batter'].unique()
            
            # Get all batters who faced the home pitcher (away team batters)
            away_batters = game_statcast[(game_statcast['pitcher'] == home_pitcher_id) & 
                                       (game_statcast['inning_topbot'] == 'Top')]['batter'].unique()
            
            # Process home team batters vs away pitcher
            for batter_id in home_batters:
                # Calculate batter stats using historical data
                batter_stats = calculate_batter_overall_stats(statcast_df_historical, batter_id)
                
                # Check if this batter hit a HR against this pitcher in this game
                batter_events = game_statcast[(game_statcast['batter'] == batter_id) & 
                                            (game_statcast['pitcher'] == away_pitcher_id)]
                hit_hr = 1 if 'home_run' in batter_events['events'].values else 0
                
                # Add matchup to the list
                all_matchups.append({
                    'game_pk': game_pk,
                    'game_date': current_date,
                    'batter_id': batter_id,
                    'pitcher_id': away_pitcher_id,
                    'batter_metrics': batter_stats,
                    'pitcher_metrics': away_pitcher_stats,
                    'ballpark_factor': ballpark_factor,
                    'weather_score': 0,  # Simplified - would need weather API integration
                    'hit_hr': hit_hr
                })
            
            # Process away team batters vs home pitcher
            for batter_id in away_batters:
                # Calculate batter stats using historical data
                batter_stats = calculate_batter_overall_stats(statcast_df_historical, batter_id)
                
                # Check if this batter hit a HR against this pitcher in this game
                batter_events = game_statcast[(game_statcast['batter'] == batter_id) & 
                                            (game_statcast['pitcher'] == home_pitcher_id)]
                hit_hr = 1 if 'home_run' in batter_events['events'].values else 0
                
                # Add matchup to the list
                all_matchups.append({
                    'game_pk': game_pk,
                    'game_date': current_date,
                    'batter_id': batter_id,
                    'pitcher_id': home_pitcher_id,
                    'batter_metrics': batter_stats,
                    'pitcher_metrics': home_pitcher_stats,
                    'ballpark_factor': ballpark_factor,
                    'weather_score': 0,  # Simplified - would need weather API integration
                    'hit_hr': hit_hr
                })
            
        current_date += timedelta(days=1)
    
    # Convert to DataFrame
    if not all_matchups:
        logger.warning("No matchups found in the specified date range")
        return pd.DataFrame()
        
    matchups_df = pd.DataFrame(all_matchups)
    logger.info(f"Retrieved {len(matchups_df)} historical matchups")
    
    return matchups_df


def backtest_model(start_date, end_date, training_window_days=365, metric_to_optimize='roc_auc'):
    """
    Backtest the model over a specific period.
    
    Args:
        start_date (datetime): Start date for backtesting
        end_date (datetime): End date for backtesting
        training_window_days (int): Number of days to use for training before each test day
        metric_to_optimize (str): Metric to optimize during weight optimization
        
    Returns:
        dict: Backtesting results with daily metrics and overall performance
    """
    logger.info(f"Starting backtesting from {start_date} to {end_date} with {training_window_days} day training window")
    
    results = []
    all_predictions = []
    all_actuals = []
    
    current_date = start_date
    while current_date <= end_date:
        # Define training period (e.g., past year)
        train_start = current_date - timedelta(days=training_window_days)
        train_end = current_date - timedelta(days=1)
        
        logger.info(f"Processing day {current_date}. Training period: {train_start} to {train_end}")
        
        # Get training data from this period
        train_data = get_historical_matchups(train_start, train_end)
        if train_data.empty:
            logger.warning(f"No training data available for {current_date}, skipping day")
            current_date += timedelta(days=1)
            continue
        
        # Optimize weights on training data
        optimized_weights, _ = optimize_weights(train_data, metric_to_optimize=metric_to_optimize)
        
        # Get actual games and matchups for current_date
        test_data = get_historical_matchups(current_date, current_date)
        if test_data.empty:
            logger.warning(f"No test data available for {current_date}, skipping day")
            current_date += timedelta(days=1)
            continue
        
        # Generate predictions for these games using optimized weights
        predictions = []
        for _, matchup in test_data.iterrows():
            score = calculate_score_with_weights(
                matchup['batter_metrics'],
                matchup['pitcher_metrics'],
                optimized_weights,
                matchup.get('ballpark_factor', 1.0),
                matchup.get('weather_score', 0)
            )
            predictions.append(score)
        
        # Record actual outcomes
        actual_outcomes = test_data['hit_hr'].tolist()
        
        # Calculate performance metrics for this day
        daily_metrics = calculate_performance_metrics(predictions, actual_outcomes)
        
        # Store results for this day
        results.append({
            'date': current_date,
            'metrics': daily_metrics,
            'num_matchups': len(test_data),
            'num_hrs': sum(actual_outcomes)
        })
        
        # Add to overall collections for aggregate metrics
        all_predictions.extend(predictions)
        all_actuals.extend(actual_outcomes)
        
        # Move to next day
        current_date += timedelta(days=1)
    
    # Calculate overall metrics
    overall_metrics = calculate_performance_metrics(all_predictions, all_actuals)
    
    logger.info(f"Backtesting complete. Overall {metric_to_optimize}: {overall_metrics.get(metric_to_optimize)}")
    
    return {
        'daily_results': results,
        'overall_metrics': overall_metrics,
        'total_matchups': len(all_actuals),
        'total_hrs': sum(all_actuals),
        'hr_rate': sum(all_actuals) / len(all_actuals) if all_actuals else 0
    }
