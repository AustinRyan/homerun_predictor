import argparse
import json
import pandas as pd
from datetime import datetime, timedelta
import logging
import sys

from algorithm.model_evaluation import (
    get_historical_matchups,
    calculate_performance_metrics
)
from algorithm.predictor import generate_hr_likelihood_score

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("simple_backtest.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def parse_date(date_str):
    """Parse date string in YYYY-MM-DD format."""
    try:
        return datetime.strptime(date_str, '%Y-%m-%d').date()
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid date format: {date_str}. Use YYYY-MM-DD.")


def simple_backtest(start_date, end_date, weights_file):
    """Run a simplified backtest using pre-optimized weights.
    
    Args:
        start_date: Start date for backtesting
        end_date: End date for backtesting
        weights_file: Path to the JSON file containing optimized weights
        
    Returns:
        dict: Backtesting results with daily metrics and overall performance
    """
    # Load optimized weights
    with open(weights_file, 'r') as f:
        weights_data = json.load(f)
    
    optimized_weights = weights_data['weights']
    logger.info(f"Loaded optimized weights from {weights_file}")
    
    # Get all matchups for the test period
    logger.info(f"Getting matchups for period {start_date} to {end_date}")
    all_matchups = get_historical_matchups(start_date, end_date)
    
    if all_matchups.empty:
        logger.error("No matchups found for the specified period.")
        return {}
    
    logger.info(f"Found {len(all_matchups)} matchups for testing")
    
    # Group matchups by date
    all_matchups['date'] = pd.to_datetime(all_matchups['game_date']).dt.date
    grouped = all_matchups.groupby('date')
    
    results = []
    all_predictions = []
    all_actuals = []
    
    # Process each day
    for date, matchups in grouped:
        logger.info(f"Processing {len(matchups)} matchups for {date}")
        
        # Generate predictions for these games using optimized weights
        predictions = []
        for _, matchup in matchups.iterrows():
            # Use the generate_hr_likelihood_score function with optimized weights
            score = generate_hr_likelihood_score(
                matchup['batter_metrics'],
                matchup['pitcher_metrics'],
                matchup.get('ballpark_factor', 1.0),
                matchup.get('weather_score', 0),
                custom_weights=optimized_weights
            )
            predictions.append(score / 100.0)  # Scale to 0-1 for metrics calculation
        
        # Record actual outcomes
        actual_outcomes = matchups['hit_hr'].tolist()
        
        # Calculate performance metrics for this day
        daily_metrics = calculate_performance_metrics(predictions, actual_outcomes)
        
        # Store results for this day
        results.append({
            'date': date,
            'metrics': daily_metrics,
            'num_matchups': len(matchups),
            'num_hrs': sum(actual_outcomes)
        })
        
        # Add to overall collections for aggregate metrics
        all_predictions.extend(predictions)
        all_actuals.extend(actual_outcomes)
    
    # Calculate overall metrics
    overall_metrics = calculate_performance_metrics(all_predictions, all_actuals)
    
    logger.info(f"Backtesting complete. Overall ROC-AUC: {overall_metrics.get('roc_auc')}")
    
    return {
        'daily_results': results,
        'overall_metrics': overall_metrics,
        'total_matchups': len(all_actuals),
        'total_hrs': sum(all_actuals),
        'hr_rate': sum(all_actuals) / len(all_actuals) if all_actuals else 0
    }


def main():
    parser = argparse.ArgumentParser(description="Run a simplified backtest using pre-optimized weights")
    parser.add_argument("--start-date", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--weights-file", required=True, help="Path to the JSON file with optimized weights")
    parser.add_argument("--output", help="Output file for results (default: simple_backtest_results_<dates>.json)")
    
    args = parser.parse_args()
    
    start_date = parse_date(args.start_date)
    end_date = parse_date(args.end_date)
    
    # Run the simplified backtest
    results = simple_backtest(start_date, end_date, args.weights_file)
    
    if not results:
        return
    
    # Save results to file
    output_file = args.output or f"simple_backtest_results_{start_date}_{end_date}.json"
    
    # Convert datetime objects to strings for JSON serialization
    serializable_results = results.copy()
    serializable_results['daily_results'] = [
        {
            **result,
            'date': result['date'].isoformat() if hasattr(result['date'], 'isoformat') else str(result['date'])
        }
        for result in results['daily_results']
    ]
    
    with open(output_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    logger.info(f"Backtesting results saved to {output_file}")
    
    # Print summary
    print("\nBacktesting Results:")
    print(f"Period: {start_date} to {end_date}")
    print(f"Total matchups evaluated: {results['total_matchups']}")
    print(f"Total home runs: {results['total_hrs']} ({results['hr_rate']*100:.2f}%)")
    print(f"Overall ROC-AUC: {results['overall_metrics'].get('roc_auc', 'N/A')}")
    print(f"Overall PR-AUC: {results['overall_metrics'].get('pr_auc', 'N/A')}")
    print(f"Overall Brier Score: {results['overall_metrics'].get('brier_score', 'N/A')}")
    print(f"Detailed results saved to: {output_file}")


if __name__ == "__main__":
    main()
