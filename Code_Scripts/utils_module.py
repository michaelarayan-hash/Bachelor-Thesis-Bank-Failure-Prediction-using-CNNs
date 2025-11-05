"""
Utility functions for bank failure prediction pipeline.
"""

import os
import logging
from typing import Dict, List
import pandas as pd


def setup_logging(log_file: str = None, level: int = logging.INFO):
    """
    Setup logging configuration.
    
    Args:
        log_file: Path to log file (None for console only)
        level: Logging level
    """
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    handlers = [logging.StreamHandler()]
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=level,
        format=log_format,
        handlers=handlers
    )


def validate_paths(paths: List[str], must_exist: bool = True):
    """
    Validate that paths exist.
    
    Args:
        paths: List of paths to validate
        must_exist: If True, raise error if path doesn't exist
        
    Raises:
        FileNotFoundError: If path doesn't exist and must_exist=True
    """
    for path in paths:
        if must_exist and not os.path.exists(path):
            raise FileNotFoundError(f"Path not found: {path}")


def create_directory_structure(base_dir: str, subdirs: List[str] = None):
    """
    Create directory structure for outputs.
    
    Args:
        base_dir: Base directory path
        subdirs: List of subdirectory names to create
    """
    os.makedirs(base_dir, exist_ok=True)
    
    if subdirs:
        for subdir in subdirs:
            os.makedirs(os.path.join(base_dir, subdir), exist_ok=True)


def merge_results(result_files: List[str], output_path: str):
    """
    Merge multiple result CSV files into one.
    
    Args:
        result_files: List of CSV file paths
        output_path: Path to save merged results
    """
    dfs = []
    for file in result_files:
        if os.path.exists(file):
            df = pd.read_csv(file)
            df['source_file'] = os.path.basename(file)
            dfs.append(df)
    
    if dfs:
        merged = pd.concat(dfs, ignore_index=True)
        merged.to_csv(output_path, index=False)
        print(f"Merged {len(dfs)} result files to: {output_path}")
    else:
        print("No result files found to merge")


def print_summary_statistics(df: pd.DataFrame, metrics: List[str]):
    """
    Print summary statistics for metrics.
    
    Args:
        df: DataFrame containing results
        metrics: List of metric column names
    """
    print("\nSummary Statistics:")
    print("=" * 60)
    for metric in metrics:
        if metric in df.columns:
            print(f"\n{metric}:")
            print(f"  Mean: {df[metric].mean():.4f}")
            print(f"  Std:  {df[metric].std():.4f}")
            print(f"  Min:  {df[metric].min():.4f}")
            print(f"  Max:  {df[metric].max():.4f}")


def compare_models(results_df: pd.DataFrame, metric: str = 'ROC AUC'):
    """
    Compare models and identify best performing.
    
    Args:
        results_df: DataFrame with results
        metric: Metric to use for comparison
        
    Returns:
        DataFrame sorted by metric
    """
    if metric not in results_df.columns:
        print(f"Metric '{metric}' not found in results")
        return results_df
    
    sorted_df = results_df.sort_values(by=metric, ascending=False)
    
    print(f"\nModel Ranking by {metric}:")
    print("=" * 60)
    for idx, row in sorted_df.iterrows():
        print(f"{row['Name']:15s}: {row[metric]:.4f}")
    
    return sorted_df