"""
Log Analysis for RL Training

This module provides tools for parsing and visualizing training logs from
reinforcement learning experiments, with a focus on symmetry-guided approaches.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
import re
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import tensorflow as tf
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def load_tensorboard_logs(log_dir: str, tag: str = 'rollout/ep_rew_mean') -> pd.DataFrame:
    """
    Load training metrics from TensorBoard logs.
    
    Args:
        log_dir: Directory containing TensorBoard logs
        tag: The specific metric tag to extract (default: episode reward mean)
        
    Returns:
        DataFrame with columns ['step', 'value']
    """
    # Find event files
    event_files = glob.glob(os.path.join(log_dir, "events.out.tfevents.*"))
    
    if not event_files:
        raise FileNotFoundError(f"No TensorBoard event files found in {log_dir}")
    
    # Load events
    data = []
    for event_file in event_files:
        event_acc = EventAccumulator(event_file)
        event_acc.Reload()
        
        # Check if the tag exists
        if tag in event_acc.Tags()['scalars']:
            # Extract values
            events = event_acc.Scalars(tag)
            for event in events:
                data.append({
                    'step': event.step,
                    'value': event.value,
                    'timestamp': event.wall_time
                })
    
    if not data:
        available_tags = event_acc.Tags()['scalars'] if 'scalars' in event_acc.Tags() else []
        raise ValueError(f"Tag '{tag}' not found in logs. Available tags: {available_tags}")
    
    # Convert to DataFrame and sort by step
    df = pd.DataFrame(data)
    df = df.sort_values('step')
    
    return df


def load_csv_logs(log_file: str) -> pd.DataFrame:
    """
    Load training metrics from CSV log files.
    
    Args:
        log_file: Path to CSV log file
        
    Returns:
        DataFrame with metrics
    """
    df = pd.read_csv(log_file)
    
    # Ensure expected columns exist
    required_cols = ['timestep', 'episode', 'reward']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        raise ValueError(f"Log file missing required columns: {missing_cols}")
    
    return df


def compute_moving_average(data: np.ndarray, window: int = 10) -> np.ndarray:
    """
    Compute moving average of a timeseries.
    
    Args:
        data: Input data array
        window: Window size for moving average
        
    Returns:
        Smoothed data array
    """
    if len(data) < window:
        return data
        
    weights = np.ones(window) / window
    return np.convolve(data, weights, mode='valid')


def plot_learning_curves(data_dict: Dict[str, pd.DataFrame], 
                         metric: str = 'reward',
                         title: str = 'Learning Curves',
                         window: int = 10,
                         x_axis: str = 'timestep',
                         output_path: Optional[str] = None,
                         show_individual_seeds: bool = False):
    """
    Plot learning curves from multiple runs.
    
    Args:
        data_dict: Dictionary mapping run names to DataFrames with metrics
        metric: Column name for the metric to plot
        title: Plot title
        window: Window size for moving average smoothing
        x_axis: Column name for x-axis values
        output_path: Path to save the visualization
        show_individual_seeds: Whether to show lines for individual seeds
    """
    plt.figure(figsize=(12, 6))
    
    for name, df in data_dict.items():
        if metric not in df.columns:
            print(f"Warning: Metric '{metric}' not found in {name} data. Skipping.")
            continue
            
        # Group by run/seed if multiple are present
        if 'seed' in df.columns:
            seeds = df['seed'].unique()
            
            # Collect data for each seed
            seed_data = []
            for seed in seeds:
                seed_df = df[df['seed'] == seed].copy()
                seed_df = seed_df.sort_values(x_axis)
                x = seed_df[x_axis].values
                y = seed_df[metric].values
                
                # Apply smoothing
                if len(y) > window:
                    y_smooth = compute_moving_average(y, window)
                    x_smooth = x[window-1:]
                else:
                    y_smooth = y
                    x_smooth = x
                
                seed_data.append((x_smooth, y_smooth))
                
                # Plot individual seed lines if requested
                if show_individual_seeds:
                    plt.plot(x_smooth, y_smooth, alpha=0.3, linewidth=0.5)
            
            # Compute mean and std across seeds
            # Interpolate to common x grid for averaging
            min_x = max([d[0][0] for d in seed_data])
            max_x = min([d[0][-1] for d in seed_data])
            x_common = np.linspace(min_x, max_x, 1000)
            
            y_interp = []
            for x_seed, y_seed in seed_data:
                y_interp.append(np.interp(x_common, x_seed, y_seed))
            
            y_mean = np.mean(y_interp, axis=0)
            y_std = np.std(y_interp, axis=0)
            
            # Plot mean with confidence interval
            plt.plot(x_common, y_mean, label=name, linewidth=2)
            plt.fill_between(x_common, y_mean - y_std, y_mean + y_std, alpha=0.2)
            
        else:
            # Single run
            df = df.sort_values(x_axis)
            x = df[x_axis].values
            y = df[metric].values
            
            # Apply smoothing
            if len(y) > window:
                y_smooth = compute_moving_average(y, window)
                x_smooth = x[window-1:]
            else:
                y_smooth = y
                x_smooth = x
                
            plt.plot(x_smooth, y_smooth, label=name, linewidth=2)
    
    plt.title(title)
    plt.xlabel(x_axis.capitalize())
    plt.ylabel(metric.capitalize())
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Save if output path is provided
    if output_path:
        # Create directory if it doesn't exist
        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Learning curve saved to {output_path}")
    
    plt.close()


def plot_seed_comparison(data_dict: Dict[str, pd.DataFrame],
                         metric: str = 'reward',
                         x_axis: str = 'timestep',
                         output_path: Optional[str] = None):
    """
    Create a grid of plots showing all seeds for each algorithm.
    
    Args:
        data_dict: Dictionary mapping run names to DataFrames with metrics
        metric: Column name for the metric to plot
        x_axis: Column name for x-axis values
        output_path: Path to save the visualization
    """
    n_methods = len(data_dict)
    
    # Determine how many seeds per method
    seeds_per_method = {}
    for name, df in data_dict.items():
        if 'seed' in df.columns:
            seeds_per_method[name] = df['seed'].nunique()
        else:
            seeds_per_method[name] = 1
    
    # Create figure with subplots
    fig, axes = plt.subplots(n_methods, 1, figsize=(12, 4 * n_methods), sharex=True)
    if n_methods == 1:
        axes = [axes]
    
    for i, (name, df) in enumerate(data_dict.items()):
        ax = axes[i]
        
        if 'seed' in df.columns:
            # Plot each seed
            for seed in df['seed'].unique():
                seed_df = df[df['seed'] == seed].copy()
                seed_df = seed_df.sort_values(x_axis)
                x = seed_df[x_axis].values
                y = seed_df[metric].values
                
                ax.plot(x, y, alpha=0.7, label=f'Seed {seed}')
        else:
            # Single run
            df = df.sort_values(x_axis)
            x = df[x_axis].values
            y = df[metric].values
            
            ax.plot(x, y, alpha=0.7, label='Run 1')
        
        ax.set_title(f'{name} - {seeds_per_method[name]} seeds')
        ax.set_ylabel(metric.capitalize())
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend()
    
    axes[-1].set_xlabel(x_axis.capitalize())
    plt.tight_layout()
    
    # Save if output path is provided
    if output_path:
        # Create directory if it doesn't exist
        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Seed comparison plot saved to {output_path}")
    
    plt.close()


def plot_final_performance_comparison(data_dict: Dict[str, pd.DataFrame],
                                     metric: str = 'reward',
                                     window: int = 10,
                                     output_path: Optional[str] = None):
    """
    Plot boxplot comparing final performance across methods.
    
    Args:
        data_dict: Dictionary mapping run names to DataFrames with metrics
        metric: Column name for the metric to plot
        window: Window size for computing final performance
        output_path: Path to save the visualization
    """
    # Collect final performance for each method and seed
    final_performances = {}
    
    for name, df in data_dict.items():
        if metric not in df.columns:
            print(f"Warning: Metric '{metric}' not found in {name} data. Skipping.")
            continue
        
        method_performances = []
        
        if 'seed' in df.columns:
            # Multiple seeds
            for seed in df['seed'].unique():
                seed_df = df[df['seed'] == seed].copy()
                # Get last window values and compute mean
                final_values = seed_df[metric].values[-window:]
                if len(final_values) > 0:
                    method_performances.append(np.mean(final_values))
        else:
            # Single run
            final_values = df[metric].values[-window:]
            if len(final_values) > 0:
                method_performances.append(np.mean(final_values))
        
        final_performances[name] = method_performances
    
    # Create boxplot
    plt.figure(figsize=(10, 6))
    
    # Collect data for boxplot
    data = [performances for name, performances in final_performances.items()]
    labels = list(final_performances.keys())
    
    # Plot
    box = plt.boxplot(data, patch_artist=True, labels=labels)
    
    # Use different colors for each box
    colors = plt.cm.tab10(np.linspace(0, 1, len(data)))
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
    
    # Add jittered points for individual seeds
    for i, (name, performances) in enumerate(final_performances.items()):
        # Add jitter for better visualization
        x = np.random.normal(i+1, 0.04, size=len(performances))
        plt.plot(x, performances, 'o', alpha=0.6, color='k', markersize=5)
    
    plt.title(f'Final {metric.capitalize()} Comparison')
    plt.ylabel(f'Final {metric.capitalize()}')
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Add mean values as text
    for i, (name, performances) in enumerate(final_performances.items()):
        if performances:
            mean_val = np.mean(performances)
            plt.text(i+1, min(performances) - (plt.ylim()[1] - plt.ylim()[0])*0.05, 
                    f'Mean: {mean_val:.2f}', ha='center')
    
    plt.tight_layout()
    
    # Save if output path is provided
    if output_path:
        # Create directory if it doesn't exist
        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Performance comparison plot saved to {output_path}")
    
    plt.close()


def analyze_symmetry_impact(log_dir: str, symmetry_info_path: str, output_dir: str):
    """
    Analyze how symmetry detection impacts learning performance.
    
    Args:
        log_dir: Directory containing training logs
        symmetry_info_path: Path to JSON file with symmetry scores
        output_dir: Directory to save analysis results
    """
    # Load symmetry information
    with open(symmetry_info_path, 'r') as f:
        symmetry_info = json.load(f)
    
    # Load training logs
    data_dict = {}
    
    # Look for runs with different baselines
    baseline_dirs = glob.glob(os.path.join(log_dir, "baseline_*"))
    for baseline_dir in baseline_dirs:
        baseline_name = os.path.basename(baseline_dir)
        
        # Load tensorboard logs
        try:
            df = load_tensorboard_logs(baseline_dir)
            data_dict[baseline_name] = df
        except (FileNotFoundError, ValueError) as e:
            print(f"Could not load logs for {baseline_name}: {str(e)}")
    
    # Plot learning curves
    if data_dict:
        plot_learning_curves(
            data_dict, 
            title='Learning Performance with Different Symmetry Baselines',
            output_path=os.path.join(output_dir, 'symmetry_impact_learning.png')
        )
        
        # Plot final performance
        plot_final_performance_comparison(
            data_dict,
            output_path=os.path.join(output_dir, 'symmetry_impact_final.png')
        )
    else:
        print("No valid baseline runs found for symmetry impact analysis.")


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description='Analyze RL training logs')
    parser.add_argument('--log_dir', type=str, required=True,
                        help='Directory containing training logs')
    parser.add_argument('--output_dir', type=str, default='visualizations',
                        help='Directory to save analysis results')
    parser.add_argument('--metric', type=str, default='reward',
                        help='Metric to analyze (e.g., reward, loss)')
    parser.add_argument('--window', type=int, default=10,
                        help='Window size for moving average')
    parser.add_argument('--tensorboard', action='store_true',
                        help='Parse TensorBoard logs instead of CSV')
    parser.add_argument('--tag', type=str, default='rollout/ep_rew_mean',
                        help='TensorBoard tag to extract (used with --tensorboard)')
    parser.add_argument('--analyze_symmetry', action='store_true',
                        help='Perform symmetry impact analysis')
    parser.add_argument('--symmetry_info', type=str, default=None,
                        help='Path to symmetry info JSON (used with --analyze_symmetry)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find logs based on specified format
    if args.tensorboard:
        # Find TensorBoard log directories
        run_dirs = [d for d in glob.glob(os.path.join(args.log_dir, "*")) 
                   if os.path.isdir(d) and os.listdir(d)]
        
        # Load data for each run
        data_dict = {}
        for run_dir in run_dirs:
            run_name = os.path.basename(run_dir)
            try:
                df = load_tensorboard_logs(run_dir, tag=args.tag)
                # Rename columns for consistency
                df = df.rename(columns={'step': 'timestep', 'value': args.metric})
                data_dict[run_name] = df
            except (FileNotFoundError, ValueError) as e:
                print(f"Could not load logs for {run_name}: {str(e)}")
                
    else:
        # Find CSV log files
        log_files = glob.glob(os.path.join(args.log_dir, "*.csv"))
        
        # Load data for each file
        data_dict = {}
        for log_file in log_files:
            run_name = os.path.splitext(os.path.basename(log_file))[0]
            try:
                df = load_csv_logs(log_file)
                data_dict[run_name] = df
            except ValueError as e:
                print(f"Could not load {log_file}: {str(e)}")
    
    if not data_dict:
        print("No valid log files found.")
        return
    
    # Generate visualizations
    print(f"Generating learning curves for {len(data_dict)} runs...")
    
    # Plot learning curves
    plot_learning_curves(
        data_dict,
        metric=args.metric,
        window=args.window,
        output_path=os.path.join(args.output_dir, 'learning_curves.png')
    )
    
    # Plot seed comparison if multiple seeds are present
    has_seeds = any('seed' in df.columns for df in data_dict.values())
    if has_seeds:
        plot_seed_comparison(
            data_dict,
            metric=args.metric,
            output_path=os.path.join(args.output_dir, 'seed_comparison.png')
        )
    
    # Plot final performance comparison
    plot_final_performance_comparison(
        data_dict,
        metric=args.metric,
        window=args.window,
        output_path=os.path.join(args.output_dir, 'final_performance.png')
    )
    
    # Perform symmetry impact analysis if requested
    if args.analyze_symmetry and args.symmetry_info:
        print("Analyzing symmetry impact...")
        analyze_symmetry_impact(
            args.log_dir,
            args.symmetry_info,
            args.output_dir
        )
    
    print(f"Analysis complete. Results saved to {args.output_dir}")


if __name__ == "__main__":
    main() 