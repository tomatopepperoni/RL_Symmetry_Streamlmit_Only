"""
Utility functions for the Streamlit app.

This module provides helper functions for data loading, processing, and visualization.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union
import os
import glob
from pathlib import Path
import json
import sys

# Add the project root to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def load_example_trajectory(name: str = "swimmer") -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load an example trajectory from the data directory.
    
    Args:
        name: Name of the example trajectory
        
    Returns:
        Tuple of (states, actions, rewards, next_states)
    """
    # Path to example data
    example_dir = Path("../data/examples")
    
    if not example_dir.exists():
        raise FileNotFoundError(f"Example directory {example_dir} not found")
    
    # Find example file
    example_file = example_dir / f"{name}_trajectory.csv"
    
    if not example_file.exists():
        # List available examples
        available_examples = [f.stem.replace("_trajectory", "") 
                            for f in example_dir.glob("*_trajectory.csv")]
        
        if available_examples:
            raise ValueError(f"Example '{name}' not found. Available examples: {available_examples}")
        else:
            raise ValueError(f"No example trajectories found in {example_dir}")
    
    # Load data using the function from symmetry_detector
    from scripts.symmetry_detector import load_trajectory_data
    return load_trajectory_data(str(example_file))


def export_results(symmetry_scores: Dict[str, float], 
                  file_path: str,
                  metadata: Optional[Dict] = None):
    """
    Export symmetry analysis results to JSON.
    
    Args:
        symmetry_scores: Dictionary of symmetry scores
        file_path: Path to save the results
        metadata: Optional metadata to include
    """
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(file_path)
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare data for export
    export_data = {
        "symmetry_scores": symmetry_scores,
    }
    
    # Add metadata if provided
    if metadata:
        export_data["metadata"] = metadata
    
    # Add timestamp
    from datetime import datetime
    export_data["timestamp"] = datetime.now().isoformat()
    
    # Save to file
    with open(file_path, 'w') as f:
        json.dump(export_data, f, indent=2)


def import_results(file_path: str) -> Dict:
    """
    Import symmetry analysis results from JSON.
    
    Args:
        file_path: Path to the results file
        
    Returns:
        Dictionary with loaded results
    """
    with open(file_path, 'r') as f:
        return json.load(f)


def create_sample_gait_plot(save_path: Optional[str] = None):
    """
    Create a sample gait plot for demonstration.
    
    Args:
        save_path: Path to save the plot image
    """
    # Create a sample figure showing an articulated swimming robot gait
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    
    # Sample data
    np.random.seed(42)
    timesteps = np.arange(0, 5, 0.05)
    
    # Plot 1: Joint angles over time
    ax = axes[0, 0]
    joint1 = 0.5 * np.sin(2 * np.pi * 0.5 * timesteps)
    joint2 = 0.5 * np.sin(2 * np.pi * 0.5 * timesteps + np.pi/2)
    ax.plot(timesteps, joint1, label='Joint 1')
    ax.plot(timesteps, joint2, label='Joint 2')
    ax.set_title("Joint Angles Over Time")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Angle (rad)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Robot trajectory
    ax = axes[0, 1]
    x = np.cumsum(0.1 * np.cos(joint1 + joint2))
    y = np.cumsum(0.1 * np.sin(joint1 + joint2))
    
    # Create a colormap to show time progression
    points = ax.scatter(x, y, c=timesteps, cmap='viridis', s=10)
    fig.colorbar(points, ax=ax, label='Time (s)')
    
    # Plot every 10th pose as a line segment to show orientation
    for i in range(0, len(timesteps), 10):
        ax.plot([x[i], x[i] + 0.2 * np.cos(joint1[i])], 
                [y[i], y[i] + 0.2 * np.sin(joint1[i])], 'r-', linewidth=1)
    
    ax.set_title("Robot Trajectory")
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    # Plot 3: Reward over time
    ax = axes[1, 0]
    rewards = 0.2 + 0.8 * (1 - np.exp(-0.5 * timesteps)) + 0.1 * np.random.randn(len(timesteps))
    ax.plot(timesteps, rewards)
    ax.set_title("Reward Over Time")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Reward")
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Symmetry visualization
    ax = axes[1, 1]
    
    # Create a radar chart for symmetry scores
    categories = ['Mirror X', 'Mirror Y', 'Time\nReversal', 'Rotation\n180°', 'Rotation\n90°']
    values = [0.92, 0.45, 0.78, 0.31, 0.15]
    
    # Convert to radians and duplicate first value for closure
    angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
    values = values + [values[0]]
    angles = angles + [angles[0]]
    categories = categories + [categories[0]]
    
    ax.plot(angles, values, 'b-', linewidth=2)
    ax.fill(angles, values, 'b', alpha=0.1)
    
    # Set category labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories[:-1])
    
    # Configure radar chart
    ax.set_ylim(0, 1)
    ax.set_title("Symmetry Scores")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save if a path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        return save_path
    
    return fig


def preprocess_trajectory_data(data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Preprocess trajectory data from a DataFrame.
    
    Args:
        data: DataFrame containing trajectory data
        
    Returns:
        Tuple of (states, actions, rewards, next_states)
    """
    # Check required columns
    state_cols = [col for col in data.columns if col.startswith('state_')]
    action_cols = [col for col in data.columns if col.startswith('action_')]
    reward_col = 'reward'
    next_state_cols = [col for col in data.columns if col.startswith('next_state_')]
    
    # Verify required columns exist
    if not state_cols:
        raise ValueError("DataFrame must contain columns prefixed with 'state_'")
    if not action_cols:
        raise ValueError("DataFrame must contain columns prefixed with 'action_'")
    if reward_col not in data.columns:
        raise ValueError("DataFrame must contain a 'reward' column")
    
    # Extract data
    states = data[state_cols].values
    actions = data[action_cols].values
    rewards = data[reward_col].values
    
    # Extract next_states if available
    if next_state_cols:
        next_states = data[next_state_cols].values
    else:
        # Create next_states by shifting states
        next_states = np.zeros_like(states)
        next_states[:-1] = states[1:]
        next_states[-1] = states[0]  # Loop around for the last step
    
    return states, actions, rewards, next_states


if __name__ == "__main__":
    # Create sample plot for documentation if run directly
    plot_path = "../visualizations/sample_gait_plot.png"
    create_sample_gait_plot(plot_path)
    print(f"Created sample plot: {plot_path}") 