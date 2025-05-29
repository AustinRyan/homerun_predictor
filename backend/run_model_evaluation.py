import argparse
import json
import pandas as pd
from datetime import datetime, timedelta
import logging
import sys

from algorithm.model_evaluation import (
    backtest_model,
    optimize_weights,
    get_historical_matchups,
    calculate_performance_metrics
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("model_evaluation.log"),
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


def run_optimization(args):
    """Run weight optimization on historical data."""
    start_date = parse_date(args.start_date)
    end_date = parse_date(args.end_date)
    
    logger.info(f"Starting weight optimization for period {start_date} to {end_date}")
    
    # Get historical matchups for the specified period
    matchups_df = get_historical_matchups(start_date, end_date)
    
    if matchups_df.empty:
        logger.error("No matchups found for the specified period.")
        return
    
    # Run optimization
    best_weights, metrics = optimize_weights(matchups_df, metric_to_optimize=args.metric)
    
    # Save results to file
    output_file = args.output or f"optimized_weights_{start_date}_{end_date}.json"
    with open(output_file, 'w') as f:
        json.dump({
            'weights': best_weights,
            'metrics': metrics,
            'period': {
                'start_date': start_date.isoformat(),
                'end_date': end_date.isoformat()
            },
            'optimization_metric': args.metric,
            'num_matchups': len(matchups_df)
        }, f, indent=2)
    
    logger.info(f"Optimization complete. Results saved to {output_file}")
    
    # Print summary
    print("\nOptimization Results:")
    print(f"Metric optimized: {args.metric}")
    print(f"Value achieved: {metrics.get(args.metric)}")
    print(f"Number of matchups: {len(matchups_df)}")
    print(f"Results saved to: {output_file}")


def run_backtesting(args):
    """Run backtesting on historical data."""
    start_date = parse_date(args.start_date)
    end_date = parse_date(args.end_date)
    
    logger.info(f"Starting backtesting for period {start_date} to {end_date}")
    
    # Run backtesting
    results = backtest_model(
        start_date, 
        end_date, 
        training_window_days=args.training_window,
        metric_to_optimize=args.metric
    )
    
    # Save results to file
    output_file = args.output or f"backtest_results_{start_date}_{end_date}.json"
    
    # Convert datetime objects to strings for JSON serialization
    serializable_results = results.copy()
    serializable_results['daily_results'] = [
        {
            **result,
            'date': result['date'].isoformat() if isinstance(result['date'], datetime) else result['date']
        }
        for result in results['daily_results']
    ]
    
    with open(output_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    logger.info(f"Backtesting complete. Results saved to {output_file}")
    
    # Print summary
    print("\nBacktesting Results:")
    print(f"Period: {start_date} to {end_date}")
    print(f"Training window: {args.training_window} days")
    print(f"Total matchups evaluated: {results['total_matchups']}")
    print(f"Total home runs: {results['total_hrs']} ({results['hr_rate']*100:.2f}%)")
    print(f"Overall {args.metric}: {results['overall_metrics'].get(args.metric)}")
    print(f"Detailed results saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Run model evaluation tasks for the homerun prediction model.")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Optimization command
    optimize_parser = subparsers.add_parser("optimize", help="Optimize model weights")
    optimize_parser.add_argument("--start-date", required=True, help="Start date (YYYY-MM-DD)")
    optimize_parser.add_argument("--end-date", required=True, help="End date (YYYY-MM-DD)")
    optimize_parser.add_argument("--metric", default="roc_auc", choices=["roc_auc", "pr_auc", "brier_score"],
                              help="Metric to optimize (default: roc_auc)")
    optimize_parser.add_argument("--output", help="Output file for results (default: optimized_weights_<dates>.json)")
    
    # Backtesting command
    backtest_parser = subparsers.add_parser("backtest", help="Backtest model performance")
    backtest_parser.add_argument("--start-date", required=True, help="Start date (YYYY-MM-DD)")
    backtest_parser.add_argument("--end-date", required=True, help="End date (YYYY-MM-DD)")
    backtest_parser.add_argument("--training-window", type=int, default=365,
                              help="Training window in days (default: 365)")
    backtest_parser.add_argument("--metric", default="roc_auc", choices=["roc_auc", "pr_auc", "brier_score"],
                              help="Metric to optimize during backtesting (default: roc_auc)")
    backtest_parser.add_argument("--output", help="Output file for results (default: backtest_results_<dates>.json)")
    
    args = parser.parse_args()
    
    if args.command == "optimize":
        run_optimization(args)
    elif args.command == "backtest":
        run_backtesting(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
