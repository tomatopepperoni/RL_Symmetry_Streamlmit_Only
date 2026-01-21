"""
Symmetry Detector for RL Trajectories

This module provides functions to detect and visualize various types of symmetry 
in reinforcement learning trajectory data.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Callable, Tuple, List, Union, Optional
import argparse
from pathlib import Path


class SymmetryDetector:
    """Class for detecting symmetry in RL trajectory data."""
    
    def __init__(self, state_dims: int, action_dims: int):
        """
        Initialize the symmetry detector.
        
        Args:
            state_dims: Number of dimensions in state space
            action_dims: Number of dimensions in action space
        """
        self.state_dims = state_dims
        self.action_dims = action_dims
        self.transformations = {}
        
        # Register default transformations
        self._register_default_transformations()
    
    def _register_default_transformations(self):
        """Register commonly used transformations for symmetry detection."""
        # Mirror symmetry transformations
        for i in range(self.state_dims):
            # Create a mirror reflection along axis i
            mirror_mask_state = np.ones(self.state_dims)
            mirror_mask_state[i] = -1
            
            # If we have velocities, they should flip too (assuming they're in the second half)
            if self.state_dims >= 2*i + 2:
                mirror_mask_state[i + self.state_dims//2] = -1
            
            # Corresponding action transformation (simplified assumption)
            mirror_mask_action = np.ones(self.action_dims)
            if i < self.action_dims:
                mirror_mask_action[i] = -1
            
            self.transformations[f"mirror_{i}"] = {
                "transform_fn": lambda s, a, ms=mirror_mask_state, ma=mirror_mask_action: (s * ms, a * ma),
                "description": f"Mirror symmetry along dimension {i}"
            }
        
        # Time reversal symmetry
        # Positions stay the same, velocities reverse
        time_mask_state = np.ones(self.state_dims)
        if self.state_dims % 2 == 0:  # Assuming second half are velocities
            time_mask_state[self.state_dims//2:] = -1
        
        self.transformations["time_reversal"] = {
            "transform_fn": lambda s, a: (s * time_mask_state, -a),
            "description": "Time reversal symmetry"
        }
        
        # 180-degree rotation (if 2D space)
        if self.state_dims >= 2:
            self.transformations["rotation_180"] = {
                "transform_fn": lambda s, a: 
                    (np.concatenate([
                        [-s[0], -s[1]], 
                        s[2:self.state_dims//2], 
                        [-s[self.state_dims//2], -s[self.state_dims//2+1]],
                        s[self.state_dims//2+2:]
                    ]) if self.state_dims >= 4 else np.array([-s[0], -s[1]]),
                     -a if self.action_dims <= 2 else np.concatenate([[-a[0], -a[1]], a[2:]])),
                "description": "180-degree rotation symmetry"
            }
    
    def register_transformation(self, name: str, transform_fn: Callable, description: str = ""):
        """
        Register a custom transformation for symmetry testing.
        
        Args:
            name: Identifier for the transformation
            transform_fn: Function that takes (state, action) and returns (transformed_state, transformed_action)
            description: Human-readable description of the transformation
        """
        self.transformations[name] = {
            "transform_fn": transform_fn,
            "description": description
        }
    
    def detect_symmetry(self, 
                        states: np.ndarray, 
                        actions: np.ndarray, 
                        rewards: np.ndarray,
                        next_states: Optional[np.ndarray] = None,
                        transformations: List[str] = None) -> Dict[str, float]:
        """
        Detect symmetry in trajectory data.
        
        Args:
            states: Array of shape (n_steps, state_dims) containing states
            actions: Array of shape (n_steps, action_dims) containing actions
            rewards: Array of shape (n_steps,) containing rewards
            next_states: Optional array of shape (n_steps, state_dims) for dynamics checking
            transformations: List of transformation names to test (default: all registered)
            
        Returns:
            Dictionary mapping transformation names to symmetry scores (0-1, higher means more symmetric)
        """
        if transformations is None:
            transformations = list(self.transformations.keys())
            
        symmetry_scores = {}
        
        for trans_name in transformations:
            if trans_name not in self.transformations:
                print(f"Warning: Transformation '{trans_name}' not registered. Skipping.")
                continue
                
            transform_fn = self.transformations[trans_name]["transform_fn"]
            
            # Apply transformation to each timestep
            transformed_states = np.zeros_like(states)
            transformed_actions = np.zeros_like(actions)
            
            for t in range(len(states)):
                transformed_states[t], transformed_actions[t] = transform_fn(states[t], actions[t])
            
            # Calculate symmetry score components
            
            # 1. State similarity (how similar are transformed states to original)
            state_diff = np.mean(np.abs(states - transformed_states)) / (np.mean(np.abs(states)) + 1e-10)
            
            # 2. Action similarity
            action_diff = np.mean(np.abs(actions - transformed_actions)) / (np.mean(np.abs(actions)) + 1e-10)
            
            # 3. Reward consistency (do symmetric states/actions yield similar rewards?)
            reward_consistency = 1.0
            if next_states is not None:
                # TODO: Check if transformed (s,a) pairs lead to expected next states and rewards
                # This requires environment dynamics and is more complex
                pass
            
            # Combine metrics (adjust weights as needed)
            symmetry_score = 1.0 - 0.5 * (state_diff + action_diff)
            
            symmetry_scores[trans_name] = symmetry_score
        
        return symmetry_scores


def load_trajectory_data(filepath: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load trajectory data from a file.
    
    Args:
        filepath: Path to trajectory data file (CSV)
        
    Returns:
        Tuple of (states, actions, rewards, next_states)
    """
    # Load data based on file extension
    if filepath.endswith('.csv'):
        df = pd.read_csv(filepath)
    elif filepath.endswith('.npy'):
        data_dict = np.load(filepath, allow_pickle=True).item()
        # Convert dict to DataFrame for consistent processing
        df = pd.DataFrame(data_dict)
    else:
        raise ValueError(f"Unsupported file format: {filepath}")
    
    # Identify column types based on naming conventions
    state_cols = [col for col in df.columns if col.startswith('state_')]
    action_cols = [col for col in df.columns if col.startswith('action_')]
    reward_col = 'reward' if 'reward' in df.columns else None
    next_state_cols = [col for col in df.columns if col.startswith('next_state_')]
    
    if not state_cols or not action_cols or not reward_col:
        raise ValueError("Data file must contain columns with prefixes 'state_', 'action_', and a 'reward' column")
    
    # Extract data
    states = df[state_cols].values
    actions = df[action_cols].values
    rewards = df[reward_col].values
    next_states = df[next_state_cols].values if next_state_cols else None
    
    return states, actions, rewards, next_states


def detect_symmetry(trajectory_path: str, output_path: Optional[str] = None) -> Dict[str, float]:
    """
    Detect symmetry in trajectory data and optionally save visualization.
    
    Args:
        trajectory_path: Path to trajectory data file
        output_path: Path to save visualization (if None, no visualization is saved)
        
    Returns:
        Dictionary of symmetry scores for different transformations
    """
    # Load trajectory data
    states, actions, rewards, next_states = load_trajectory_data(trajectory_path)
    
    # Initialize symmetry detector
    detector = SymmetryDetector(state_dims=states.shape[1], action_dims=actions.shape[1])
    
    # Detect symmetry
    symmetry_scores = detector.detect_symmetry(states, actions, rewards, next_states)
    
    # Visualize if output path is provided
    if output_path:
        plot_symmetry_scores(symmetry_scores, output_path)
    
    return symmetry_scores


def plot_symmetry_scores(symmetry_scores: Dict[str, float], output_path: Optional[str] = None):
    """
    Visualize symmetry scores.
    
    Args:
        symmetry_scores: Dictionary mapping transformation names to symmetry scores
        output_path: Path to save the visualization (if None, just displays the plot)
    """
    plt.figure(figsize=(10, 6))
    
    # Sort transformations by score for better visualization
    sorted_items = sorted(symmetry_scores.items(), key=lambda x: x[1], reverse=True)
    trans_names = [item[0] for item in sorted_items]
    scores = [item[1] for item in sorted_items]
    
    # Create bar plot
    bars = plt.bar(trans_names, scores)
    
    # Customize plot
    plt.ylim(0, 1)
    plt.title('Symmetry Scores for Different Transformations')
    plt.ylabel('Symmetry Score (higher is more symmetric)')
    plt.xlabel('Transformation Type')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(rotation=45, ha='right')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.2f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    # Save if output path is provided
    if output_path:
        # Create directory if it doesn't exist
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Symmetry visualization saved to {output_path}")
    
    plt.close()


def visualize_transformation(states: np.ndarray, 
                             transformed_states: np.ndarray, 
                             transformation_name: str,
                             output_path: Optional[str] = None):
    """
    Visualize original and transformed trajectories.
    
    Args:
        states: Original state trajectory
        transformed_states: Transformed state trajectory
        transformation_name: Name of the transformation
        output_path: Path to save the visualization
    """
    if states.shape[1] < 2:
        print("Warning: Cannot visualize trajectory with less than 2 dimensions")
        return
    
    plt.figure(figsize=(12, 6))
    
    # Plot original trajectory
    plt.subplot(1, 2, 1)
    plt.plot(states[:, 0], states[:, 1], 'b-', linewidth=1.5)
    plt.title('Original Trajectory')
    plt.xlabel('Dimension 0')
    plt.ylabel('Dimension 1')
    plt.grid(True, alpha=0.3)
    
    # Plot transformed trajectory
    plt.subplot(1, 2, 2)
    plt.plot(transformed_states[:, 0], transformed_states[:, 1], 'r-', linewidth=1.5)
    plt.title(f'After {transformation_name}')
    plt.xlabel('Dimension 0')
    plt.ylabel('Dimension 1')
    plt.grid(True, alpha=0.3)
    
    plt.suptitle(f'Trajectory Transformation: {transformation_name}')
    plt.tight_layout()
    
    if output_path:
        # Create directory if it doesn't exist
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Transformation visualization saved to {output_path}")
    
    plt.close()


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description='Detect symmetry in RL trajectory data')
    parser.add_argument('--input_path', type=str, required=True,
                        help='Path to trajectory data file (CSV or NPY)')
    parser.add_argument('--output_path', type=str, default=None,
                        help='Path to save symmetry visualization')
    parser.add_argument('--vis_transformation', type=str, default=None,
                        help='Visualize specific transformation (e.g., "mirror_0")')
    
    args = parser.parse_args()
    
    # Load trajectory data
    states, actions, rewards, next_states = load_trajectory_data(args.input_path)
    
    # Initialize symmetry detector
    detector = SymmetryDetector(state_dims=states.shape[1], action_dims=actions.shape[1])
    
    # Detect symmetry
    symmetry_scores = detector.detect_symmetry(states, actions, rewards, next_states)
    
    # Print results
    print("\nSymmetry Detection Results:")
    print("--------------------------")
    for trans_name, score in sorted(symmetry_scores.items(), key=lambda x: x[1], reverse=True):
        description = detector.transformations[trans_name]["description"]
        print(f"{trans_name} ({description}): {score:.4f}")
    
    # Visualize scores
    if args.output_path:
        plot_symmetry_scores(symmetry_scores, args.output_path)
    
    # Visualize specific transformation if requested
    if args.vis_transformation and args.vis_transformation in detector.transformations:
        trans_name = args.vis_transformation
        transform_fn = detector.transformations[trans_name]["transform_fn"]
        
        # Apply transformation
        transformed_states = np.zeros_like(states)
        transformed_actions = np.zeros_like(actions)
        
        for t in range(len(states)):
            transformed_states[t], transformed_actions[t] = transform_fn(states[t], actions[t])
        
        # Create output path for transformation visualization
        if args.output_path:
            output_dir = Path(args.output_path).parent
            trans_vis_path = output_dir / f"transformation_{trans_name}.png"
        else:
            trans_vis_path = None
        
        visualize_transformation(states, transformed_states, trans_name, trans_vis_path)


if __name__ == "__main__":
    main() 