#!/usr/bin/env python3
"""
Generate sample trajectory data for RL Symmetry Analysis Dashboard
"""
import numpy as np
import pandas as pd
from pathlib import Path

def generate_biped_trajectory(timesteps=500, noise_level=0.1, name="experiment_1"):
    """
    Generate synthetic biped locomotion trajectory with symmetry patterns
    
    Args:
        timesteps: Number of time steps
        noise_level: Amount of noise to add (0.0 to 1.0)
        name: Dataset name
    """
    np.random.seed(42)
    
    # Time array
    t = np.linspace(0, 10, timesteps)
    
    # Generate periodic patterns (mimicking leg movements)
    freq = 2.0  # Hz
    
    # Left leg (x, y, z positions)
    left_hip_x = 0.1 * np.sin(2 * np.pi * freq * t) + noise_level * np.random.randn(timesteps) * 0.05
    left_hip_y = 0.3 + 0.05 * np.cos(2 * np.pi * freq * t) + noise_level * np.random.randn(timesteps) * 0.03
    left_hip_z = 0.5 + 0.1 * np.abs(np.sin(2 * np.pi * freq * t)) + noise_level * np.random.randn(timesteps) * 0.02
    
    # Right leg (mirrored with slight phase shift for realistic gait)
    phase_shift = np.pi  # 180 degrees out of phase
    right_hip_x = -0.1 * np.sin(2 * np.pi * freq * t + phase_shift) + noise_level * np.random.randn(timesteps) * 0.05
    right_hip_y = 0.3 + 0.05 * np.cos(2 * np.pi * freq * t + phase_shift) + noise_level * np.random.randn(timesteps) * 0.03
    right_hip_z = 0.5 + 0.1 * np.abs(np.sin(2 * np.pi * freq * t + phase_shift)) + noise_level * np.random.randn(timesteps) * 0.02
    
    # Torso (center of mass)
    torso_x = 0.02 * np.sin(4 * np.pi * freq * t) + noise_level * np.random.randn(timesteps) * 0.01
    torso_y = 0.5 + 0.03 * np.cos(2 * np.pi * freq * t) + noise_level * np.random.randn(timesteps) * 0.02
    torso_z = 1.0 + 0.02 * np.sin(4 * np.pi * freq * t) + noise_level * np.random.randn(timesteps) * 0.01
    
    # Velocities (derivatives)
    left_hip_vx = np.gradient(left_hip_x, t)
    left_hip_vy = np.gradient(left_hip_y, t)
    left_hip_vz = np.gradient(left_hip_z, t)
    
    right_hip_vx = np.gradient(right_hip_x, t)
    right_hip_vy = np.gradient(right_hip_y, t)
    right_hip_vz = np.gradient(right_hip_z, t)
    
    # Reward (higher when more symmetric)
    symmetry_error = np.sqrt((left_hip_x + right_hip_x)**2 + 
                             (left_hip_z - right_hip_z)**2)
    reward = 1.0 - 0.5 * symmetry_error + noise_level * np.random.randn(timesteps) * 0.1
    
    # Policy loss (decreasing over time with noise)
    policy_loss = 1.0 * np.exp(-t / 5.0) + noise_level * np.random.randn(timesteps) * 0.05
    policy_loss = np.maximum(policy_loss, 0.01)  # Keep positive
    
    # Create DataFrame
    df = pd.DataFrame({
        'timestep': np.arange(timesteps),
        'time': t,
        'left_hip_x': left_hip_x,
        'left_hip_y': left_hip_y,
        'left_hip_z': left_hip_z,
        'left_hip_vx': left_hip_vx,
        'left_hip_vy': left_hip_vy,
        'left_hip_vz': left_hip_vz,
        'right_hip_x': right_hip_x,
        'right_hip_y': right_hip_y,
        'right_hip_z': right_hip_z,
        'right_hip_vx': right_hip_vx,
        'right_hip_vy': right_hip_vy,
        'right_hip_vz': right_hip_vz,
        'torso_x': torso_x,
        'torso_y': torso_y,
        'torso_z': torso_z,
        'reward': reward,
        'policy_loss': policy_loss,
    })
    
    return df

def generate_multiple_experiments():
    """Generate multiple experiment datasets with varying characteristics"""
    
    output_dir = Path(__file__).parent.parent / "data"
    output_dir.mkdir(exist_ok=True)
    
    # Experiment 1: Low noise, good symmetry
    df1 = generate_biped_trajectory(timesteps=500, noise_level=0.1, name="experiment_1")
    df1.to_csv(output_dir / "experiment_1.csv", index=False)
    print(f"✅ Generated: experiment_1.csv ({len(df1)} timesteps)")
    
    # Experiment 2: Higher noise, learning phase
    df2 = generate_biped_trajectory(timesteps=600, noise_level=0.3, name="experiment_2")
    df2.to_csv(output_dir / "experiment_2.csv", index=False)
    print(f"✅ Generated: experiment_2.csv ({len(df2)} timesteps)")
    
    # Experiment 3: Very smooth, converged policy
    df3 = generate_biped_trajectory(timesteps=400, noise_level=0.05, name="experiment_3")
    df3.to_csv(output_dir / "experiment_3.csv", index=False)
    print(f"✅ Generated: experiment_3.csv ({len(df3)} timesteps)")
    
    print(f"\n📂 Saved to: {output_dir.absolute()}")
    print("\n📊 Data columns:", list(df1.columns))
    print(f"\n💡 Use these files in the Streamlit dashboard!")

if __name__ == "__main__":
    generate_multiple_experiments()

