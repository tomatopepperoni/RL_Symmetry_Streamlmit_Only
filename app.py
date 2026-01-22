"""
Advanced Streamlit Dashboard for RL Symmetry Visualization
Enhanced version with full interactivity and user controls
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import sys
import json
from pathlib import Path
from datetime import datetime
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
import io
import seaborn as sns

# Add the project root to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.symmetry_detector import SymmetryDetector, load_trajectory_data

# Define compute_moving_average directly (no need to import from analyze_logs)
def compute_moving_average(data: np.ndarray, window: int = 10) -> np.ndarray:
    """Compute moving average of a timeseries."""
    if len(data) < window:
        return data
    weights = np.ones(window) / window
    return np.convolve(data, weights, mode='valid')


# Page configuration
st.set_page_config(
    page_title="RL Symmetry Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize session state variables and load default demo data."""
    if 'datasets' not in st.session_state:
        st.session_state.datasets = {}  # Multiple datasets
    if 'current_dataset' not in st.session_state:
        st.session_state.current_dataset = None
    if 'symmetry_results' not in st.session_state:
        st.session_state.symmetry_results = {}
    if 'comparison_mode' not in st.session_state:
        st.session_state.comparison_mode = False
    if 'analysis_cache' not in st.session_state:
        st.session_state.analysis_cache = {}
    
    # Auto-load demo data on first visit
    if 'initialized' not in st.session_state:
        st.session_state.initialized = True
        
        # Generate default demo data
        states, actions, rewards, next_states = create_demo_data(
            timesteps=1000, 
            dim_state=8, 
            dim_action=2, 
            noise_level=0.15
        )
        
        st.session_state.datasets["demo_biped_locomotion"] = {
            "states": states,
            "actions": actions,
            "rewards": rewards,
            "next_states": next_states,
            "source": "auto_demo",
            "timestamp": datetime.now(),
            "params": {
                "timesteps": 1000,
                "state_dim": 8,
                "action_dim": 2,
                "noise": 0.15,
                "description": "Biped locomotion with realistic gait patterns"
            }
        }
        st.session_state.current_dataset = "demo_biped_locomotion"


def load_sample_csv_data(csv_path):
    """Load sample CSV data from the data/ directory."""
    try:
        df = pd.read_csv(csv_path)
        
        # Extract states (left/right hip, torso positions)
        state_cols = [c for c in df.columns if any(x in c for x in ['left_hip', 'right_hip', 'torso']) and '_v' not in c]
        states = df[state_cols].values if state_cols else df[['time']].values
        
        # Extract actions (velocities)
        action_cols = [c for c in df.columns if '_v' in c]
        actions = df[action_cols].values if action_cols else np.zeros((len(df), 1))
        
        # Extract rewards
        rewards = df['reward'].values if 'reward' in df.columns else np.ones(len(df))
        
        # Create next_states
        next_states = np.roll(states, -1, axis=0)
        next_states[-1] = states[-1]
        
        return states, actions, rewards, next_states
    except Exception as e:
        st.error(f"Error loading CSV: {str(e)}")
        return None, None, None, None


def create_demo_data(timesteps=1000, dim_state=8, dim_action=2, noise_level=0.1):
    """
    Create realistic synthetic biped locomotion data with rich patterns.
    
    Args:
        timesteps: Number of timesteps
        dim_state: State dimension (minimum 8 for biped)
        dim_action: Action dimension
        noise_level: Amount of random noise (0-1)
    """
    np.random.seed(42)
    
    # Time array
    t_array = np.linspace(0, 10, timesteps)
    
    # Create rich state patterns (mimicking biped robot)
    states = np.zeros((timesteps, dim_state))
    
    # Dimension 0-1: Hip positions with walking gait (sinusoidal)
    frequency = 2.0  # Hz
    states[:, 0] = 0.3 * np.sin(2 * np.pi * frequency * t_array) + noise_level * np.random.randn(timesteps) * 0.05
    states[:, 1] = 0.3 * np.cos(2 * np.pi * frequency * t_array) + noise_level * np.random.randn(timesteps) * 0.05
    
    # Dimension 2-3: Knee angles with phase shift
    states[:, 2] = 0.4 * np.sin(2 * np.pi * frequency * t_array + np.pi/4) + noise_level * np.random.randn(timesteps) * 0.03
    states[:, 3] = 0.4 * np.cos(2 * np.pi * frequency * t_array + np.pi/4) + noise_level * np.random.randn(timesteps) * 0.03
    
    # Dimension 4-5: Torso position and orientation
    states[:, 4] = 0.1 * np.sin(4 * np.pi * frequency * t_array) + noise_level * np.random.randn(timesteps) * 0.02
    states[:, 5] = 1.0 + 0.05 * np.cos(2 * np.pi * frequency * t_array) + noise_level * np.random.randn(timesteps) * 0.01
    
    # Dimension 6-7: Angular velocities
    if dim_state > 6:
        states[:, 6] = np.gradient(states[:, 0], t_array) + noise_level * np.random.randn(timesteps) * 0.01
        states[:, 7] = np.gradient(states[:, 1], t_array) + noise_level * np.random.randn(timesteps) * 0.01
    
    # Additional dimensions if needed
    for i in range(8, dim_state):
        states[:, i] = 0.1 * noise_level * np.random.randn(timesteps)
    
    # Create actions correlated with states
    actions = np.zeros((timesteps, dim_action))
    actions[:, 0] = 0.5 * np.sin(2 * np.pi * frequency * t_array + np.pi/6) + noise_level * np.random.randn(timesteps) * 0.05
    if dim_action > 1:
        actions[:, 1] = 0.5 * np.cos(2 * np.pi * frequency * t_array - np.pi/6) + noise_level * np.random.randn(timesteps) * 0.05
    
    # Add more actions if needed
    for i in range(2, dim_action):
        actions[:, i] = 0.3 * np.sin(2 * np.pi * frequency * t_array + i) + noise_level * np.random.randn(timesteps) * 0.05
    
    # Rewards: Higher when movement is smooth and symmetric
    # Calculate smoothness (inverse of acceleration magnitude)
    velocity = np.diff(states[:, :2], axis=0)
    velocity_mag = np.linalg.norm(velocity, axis=1)
    smoothness = 1.0 / (1.0 + np.abs(np.diff(velocity_mag)))
    smoothness = np.concatenate([[smoothness[0]], smoothness, [smoothness[-1]]])  # Pad to match timesteps
    
    # Calculate symmetry (similarity between left and right)
    symmetry = 1.0 - np.abs(states[:, 0] + states[:, 1]) / 2.0
    
    # Combine factors for reward
    rewards = 0.5 * smoothness + 0.3 * symmetry + 0.2 * np.sin(2 * np.pi * frequency * t_array / 2)
    rewards = np.maximum(rewards, 0.0)  # Keep positive
    rewards += noise_level * np.random.randn(timesteps) * 0.1
    rewards = np.clip(rewards, 0.0, 1.0)
    
    # Create next_states
    next_states = np.roll(states, -1, axis=0)
    next_states[-1] = states[-1]  # Last state transitions to itself
    
    return states, actions, rewards, next_states


def plot_trajectory_interactive(states, selected_dims, title="Agent Trajectory", color_by_time=True):
    """Interactive trajectory plot with dimension selection."""
    if len(selected_dims) < 2:
        st.warning("Please select at least 2 dimensions")
        return None
    
    dim_x, dim_y = selected_dims[:2]
    
    if color_by_time:
        fig = px.line(x=states[:, dim_x], y=states[:, dim_y],
                      color=np.arange(len(states)),
                      title=title,
                      labels={"x": f"Dimension {dim_x}", "y": f"Dimension {dim_y}", "color": "Time"})
    else:
        fig = px.line(x=states[:, dim_x], y=states[:, dim_y],
                      title=title,
                      labels={"x": f"Dimension {dim_x}", "y": f"Dimension {dim_y}"})
    
    fig.update_layout(hovermode="closest", height=500)
    return fig


def plot_rewards_interactive(rewards, window_size, show_raw, show_smoothed, show_cumulative):
    """Interactive reward plot with multiple display options."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    timesteps = np.arange(len(rewards))
    
    if show_raw:
        fig.add_trace(
            go.Scatter(x=timesteps, y=rewards, mode='lines', name='Raw Rewards',
                      line=dict(color='rgba(0,100,255,0.3)', width=1)),
            secondary_y=False
        )
    
    if show_smoothed and len(rewards) > window_size:
        smoothed = compute_moving_average(rewards, window_size)
        smoothed_steps = timesteps[window_size-1:]
        fig.add_trace(
            go.Scatter(x=smoothed_steps, y=smoothed, mode='lines', name=f'Smoothed (w={window_size})',
                      line=dict(color='blue', width=2)),
            secondary_y=False
        )
    
    if show_cumulative:
        fig.add_trace(
            go.Scatter(x=timesteps, y=np.cumsum(rewards), mode='lines', name='Cumulative',
                      line=dict(color='green', width=2)),
            secondary_y=True
        )
    
    fig.update_layout(title='Reward Analysis', xaxis_title='Timestep', hovermode="x unified", height=500)
    fig.update_yaxes(title_text="Reward", secondary_y=False)
    fig.update_yaxes(title_text="Cumulative Reward", secondary_y=True)
    
    return fig


def analyze_symmetry_interactive(states, actions, rewards, selected_transforms, custom_threshold=0.7):
    """Interactive symmetry analysis with user-selected transformations."""
    detector = SymmetryDetector(state_dims=states.shape[1], action_dims=actions.shape[1])
    
    # Filter transformations
    if selected_transforms:
        available_transforms = [t for t in detector.transformations.keys() if t in selected_transforms]
    else:
        available_transforms = list(detector.transformations.keys())
    
    symmetry_scores = detector.detect_symmetry(states, actions, rewards, transformations=available_transforms)
    
    # Store transformed data
    transformed_data = {}
    for trans_name in symmetry_scores.keys():
        transform_fn = detector.transformations[trans_name]["transform_fn"]
        transformed_states = np.zeros_like(states)
        transformed_actions = np.zeros_like(actions)
        
        for t in range(len(states)):
            transformed_states[t], transformed_actions[t] = transform_fn(states[t], actions[t])
        
        transformed_data[trans_name] = {
            "states": transformed_states,
            "actions": transformed_actions,
            "score": symmetry_scores[trans_name]
        }
    
    return symmetry_scores, transformed_data


def cluster_behaviors_interactive(states, actions, n_clusters, algorithm, n_components=2):
    """Interactive behavior clustering with algorithm selection."""
    # Combine states and actions
    X = np.hstack([states, actions])
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    
    # Clustering
    if algorithm == "K-Means":
        clusterer = KMeans(n_clusters=n_clusters, random_state=42)
        labels = clusterer.fit_predict(X_pca)
        centers = clusterer.cluster_centers_
    elif algorithm == "DBSCAN":
        clusterer = DBSCAN(eps=0.5, min_samples=5)
        labels = clusterer.fit_predict(X_pca)
        centers = None
    else:
        labels = np.zeros(len(X_pca))
        centers = None
    
    return X_pca, labels, centers, pca.explained_variance_ratio_


def compare_datasets(datasets_dict, metric='reward', window=10):
    """Compare multiple datasets."""
    fig = go.Figure()
    
    for name, data in datasets_dict.items():
        rewards = data['rewards']
        timesteps = np.arange(len(rewards))
        
        if len(rewards) > window:
            smoothed = compute_moving_average(rewards, window)
            smoothed_steps = timesteps[window-1:]
            fig.add_trace(go.Scatter(x=smoothed_steps, y=smoothed, mode='lines', name=name))
    
    fig.update_layout(title='Dataset Comparison', xaxis_title='Timestep', yaxis_title='Reward', height=500)
    return fig


def plot_state_action_correlation(states, actions):
    """Create correlation heatmap between states and actions."""
    # Combine states and actions
    combined = np.hstack([states, actions])
    n_states = states.shape[1]
    n_actions = actions.shape[1]
    
    # Calculate correlation
    corr_matrix = np.corrcoef(combined.T)
    
    # Extract state-action correlation block
    state_action_corr = corr_matrix[:n_states, n_states:]
    
    # Create labels
    state_labels = [f'S{i}' for i in range(n_states)]
    action_labels = [f'A{i}' for i in range(n_actions)]
    
    fig = go.Figure(data=go.Heatmap(
        z=state_action_corr,
        x=action_labels,
        y=state_labels,
        colorscale='RdBu',
        zmid=0,
        text=np.round(state_action_corr, 2),
        texttemplate='%{text}',
        textfont={"size": 10},
        colorbar=dict(title="Correlation")
    ))
    
    fig.update_layout(
        title='State-Action Correlation Matrix',
        xaxis_title='Actions',
        yaxis_title='States',
        height=400
    )
    return fig


def plot_trajectory_3d(states, color_by_time=True):
    """Create 3D trajectory visualization."""
    if states.shape[1] < 3:
        return None
    
    if color_by_time:
        fig = go.Figure(data=[go.Scatter3d(
            x=states[:, 0],
            y=states[:, 1],
            z=states[:, 2],
            mode='lines+markers',
            marker=dict(
                size=3,
                color=np.arange(len(states)),
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Time")
            ),
            line=dict(color='rgba(100,100,100,0.3)', width=2)
        )])
    else:
        fig = go.Figure(data=[go.Scatter3d(
            x=states[:, 0],
            y=states[:, 1],
            z=states[:, 2],
            mode='lines',
            line=dict(color='blue', width=3)
        )])
    
    fig.update_layout(
        title='3D Trajectory Visualization',
        scene=dict(
            xaxis_title='Dimension 0',
            yaxis_title='Dimension 1',
            zaxis_title='Dimension 2'
        ),
        height=600
    )
    return fig


def plot_velocity_acceleration(states):
    """Analyze velocity and acceleration from state changes."""
    # Calculate velocity (first derivative)
    velocity = np.diff(states, axis=0)
    velocity_magnitude = np.linalg.norm(velocity, axis=1)
    
    # Calculate acceleration (second derivative)
    acceleration = np.diff(velocity, axis=0)
    acceleration_magnitude = np.linalg.norm(acceleration, axis=1)
    
    fig = make_subplots(rows=2, cols=1, subplot_titles=("Velocity Magnitude", "Acceleration Magnitude"))
    
    fig.add_trace(
        go.Scatter(x=np.arange(len(velocity_magnitude)), y=velocity_magnitude, 
                   mode='lines', name='Velocity', line=dict(color='blue')),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=np.arange(len(acceleration_magnitude)), y=acceleration_magnitude,
                   mode='lines', name='Acceleration', line=dict(color='red')),
        row=2, col=1
    )
    
    fig.update_xaxes(title_text="Timestep", row=2, col=1)
    fig.update_yaxes(title_text="Magnitude", row=1, col=1)
    fig.update_yaxes(title_text="Magnitude", row=2, col=1)
    fig.update_layout(height=600, showlegend=False)
    
    return fig, velocity_magnitude, acceleration_magnitude


def plot_action_energy(actions):
    """Calculate and plot action energy (magnitude)."""
    action_energy = np.linalg.norm(actions, axis=1)
    cumulative_energy = np.cumsum(action_energy)
    
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Action Energy", "Cumulative Energy"))
    
    fig.add_trace(
        go.Scatter(x=np.arange(len(action_energy)), y=action_energy,
                   mode='lines', name='Energy', line=dict(color='purple')),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=np.arange(len(cumulative_energy)), y=cumulative_energy,
                   mode='lines', name='Cumulative', line=dict(color='orange')),
        row=1, col=2
    )
    
    fig.update_xaxes(title_text="Timestep")
    fig.update_yaxes(title_text="Energy")
    fig.update_layout(height=400, showlegend=False)
    
    return fig, action_energy


def generate_statistics_report(states, actions, rewards):
    """Generate comprehensive statistics report."""
    stats = {
        "States": {
            "Mean": np.mean(states, axis=0),
            "Std": np.std(states, axis=0),
            "Min": np.min(states, axis=0),
            "Max": np.max(states, axis=0),
        },
        "Actions": {
            "Mean": np.mean(actions, axis=0),
            "Std": np.std(actions, axis=0),
            "Min": np.min(actions, axis=0),
            "Max": np.max(actions, axis=0),
        },
        "Rewards": {
            "Total": np.sum(rewards),
            "Mean": np.mean(rewards),
            "Std": np.std(rewards),
            "Min": np.min(rewards),
            "Max": np.max(rewards),
            "Median": np.median(rewards),
        }
    }
    return stats


def main():
    """Main dashboard application."""
    init_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">🤖 RL Symmetry Analysis Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar - Data Management
    with st.sidebar:
        st.header("📁 Data Management")
        
        # Dataset selection
        if st.session_state.datasets:
            dataset_names = list(st.session_state.datasets.keys())
            selected_dataset = st.selectbox("Select Dataset", dataset_names)
            st.session_state.current_dataset = selected_dataset
            
            # Show if it's the default demo
            if selected_dataset == "demo_biped_locomotion":
                st.success("✅ Default demo data loaded")
        else:
            st.info("No datasets loaded. Upload or generate data below.")
            selected_dataset = None
        
        st.markdown("---")
        
        # Data loading options
        data_source = st.radio("Data Source", ["Upload CSV", "Generate Demo Data", "Compare Multiple"])
        
        if data_source == "Upload CSV":
            uploaded_file = st.file_uploader("Upload trajectory data (CSV)", type=["csv"])
            dataset_name = st.text_input("Dataset Name", "experiment_1")
            
            if uploaded_file and st.button("Load Data"):
                try:
                    temp_dir = Path("temp_uploads")
                    temp_dir.mkdir(exist_ok=True)
                    temp_file = temp_dir / f"{dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                    
                    with open(temp_file, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    states, actions, rewards, next_states = load_trajectory_data(str(temp_file))
                    
                    st.session_state.datasets[dataset_name] = {
                        "states": states,
                        "actions": actions,
                        "rewards": rewards,
                        "next_states": next_states,
                        "source": str(temp_file),
                        "timestamp": datetime.now()
                    }
                    st.session_state.current_dataset = dataset_name
                    st.success(f"✅ Loaded {len(states)} timesteps")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
        
        elif data_source == "Generate Demo Data":
            demo_type = st.radio("Demo Data Type", ["🤖 Synthetic Pattern", "📁 Load Sample CSV"])
            
            if demo_type == "📁 Load Sample CSV":
                # Check for available CSV files
                data_dir = Path("data")
                if data_dir.exists():
                    csv_files = list(data_dir.glob("*.csv"))
                    if csv_files:
                        selected_csv = st.selectbox("Select Sample CSV", [f.name for f in csv_files])
                        dataset_name = st.text_input("Dataset Name", selected_csv.replace('.csv', ''))
                        
                        if st.button("Load Sample CSV"):
                            csv_path = data_dir / selected_csv
                            states, actions, rewards, next_states = load_sample_csv_data(csv_path)
                            
                            if states is not None:
                                st.session_state.datasets[dataset_name] = {
                                    "states": states,
                                    "actions": actions,
                                    "rewards": rewards,
                                    "next_states": next_states,
                                    "source": str(csv_path),
                                    "timestamp": datetime.now()
                                }
                                st.session_state.current_dataset = dataset_name
                                st.success(f"✅ Loaded {len(states)} timesteps from {selected_csv}")
                                st.rerun()
                    else:
                        st.warning("No CSV files found in data/ directory")
                else:
                    st.warning("data/ directory not found")
            
            else:  # Synthetic Pattern
                st.subheader("Demo Data Parameters")
                demo_timesteps = st.slider("Timesteps", 100, 2000, 1000, 100)
                demo_state_dim = st.slider("State Dimension", 2, 16, 8, 2)
                demo_action_dim = st.slider("Action Dimension", 1, 8, 2, 1)
                demo_noise = st.slider("Noise Level", 0.0, 1.0, 0.1, 0.05)
                
                dataset_name = st.text_input("Dataset Name", "demo_data")
                
                if st.button("Generate Data"):
                    states, actions, rewards, next_states = create_demo_data(
                        demo_timesteps, demo_state_dim, demo_action_dim, demo_noise
                    )
                    
                    st.session_state.datasets[dataset_name] = {
                        "states": states,
                        "actions": actions,
                        "rewards": rewards,
                        "next_states": next_states,
                        "source": "demo",
                        "timestamp": datetime.now(),
                        "params": {
                            "timesteps": demo_timesteps,
                            "state_dim": demo_state_dim,
                            "action_dim": demo_action_dim,
                            "noise": demo_noise
                        }
                    }
                    st.session_state.current_dataset = dataset_name
                    st.success(f"✅ Generated {demo_timesteps} timesteps")
                    st.rerun()
        
        elif data_source == "Compare Multiple":
            st.session_state.comparison_mode = True
            st.info("Comparison mode enabled. Upload or generate multiple datasets to compare.")
        
        st.markdown("---")
        
        # Dataset management
        if st.session_state.datasets:
            st.subheader("Loaded Datasets")
            for name in list(st.session_state.datasets.keys()):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.text(f"📊 {name}")
                with col2:
                    if st.button("🗑️", key=f"del_{name}"):
                        del st.session_state.datasets[name]
                        if st.session_state.current_dataset == name:
                            st.session_state.current_dataset = None
                        st.rerun()
    
    # Main content area
    if not st.session_state.datasets:
        st.info("⚠️ No datasets loaded. Reloading page will restore default demo data.")
        return
    
    # Get current dataset
    current_data = st.session_state.datasets.get(st.session_state.current_dataset)
    
    if current_data is None:
        st.warning("Please select a dataset from the sidebar.")
        return
    
    states = current_data["states"]
    actions = current_data["actions"]
    rewards = current_data["rewards"]
    next_states = current_data["next_states"]
    
    # Dataset info
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📊 Dataset", st.session_state.current_dataset)
    with col2:
        st.metric("⏱️ Timesteps", len(states))
    with col3:
        st.metric("📐 State Dim", states.shape[1])
    with col4:
        st.metric("🎮 Action Dim", actions.shape[1])
    
    st.markdown("---")
    
    # Tabs
    if st.session_state.comparison_mode and len(st.session_state.datasets) > 1:
        tabs = st.tabs(["📊 Comparison", "🎯 Trajectory", "💰 Rewards", "🔄 Symmetry", "🧩 Clustering", "🔬 Advanced", "ℹ️ About"])
        
        with tabs[0]:
            st.header("Dataset Comparison")
            st.markdown("""
            **Compare multiple datasets side-by-side**
            - Overlay reward curves
            - Compare symmetry scores
            - Identify best performing strategies
            """)
            
            selected_datasets = st.multiselect(
                "Select datasets to compare",
                list(st.session_state.datasets.keys()),
                default=list(st.session_state.datasets.keys())[:3]
            )
            
            if selected_datasets:
                datasets_to_compare = {name: st.session_state.datasets[name] for name in selected_datasets}
                
                # Reward comparison
                comp_window = st.slider("Smoothing window", 1, 100, 10, key="comp_window")
                fig_comp = compare_datasets(datasets_to_compare, window=comp_window)
                st.plotly_chart(fig_comp, use_container_width=True)
                
                # Summary statistics
                st.subheader("Summary Statistics")
                summary_data = []
                for name, data in datasets_to_compare.items():
                    summary_data.append({
                        "Dataset": name,
                        "Mean Reward": f"{np.mean(data['rewards']):.3f}",
                        "Max Reward": f"{np.max(data['rewards']):.3f}",
                        "Total Reward": f"{np.sum(data['rewards']):.3f}",
                        "Timesteps": len(data['rewards'])
                    })
                st.dataframe(pd.DataFrame(summary_data), use_container_width=True)
        
        tab_offset = 1
    else:
        tabs = st.tabs(["🎯 Trajectory", "💰 Rewards", "🔄 Symmetry", "🧩 Clustering", "🔬 Advanced", "ℹ️ About"])
        tab_offset = 0
    
    # Trajectory Analysis
    with tabs[tab_offset]:
        st.header("Trajectory Analysis")
        st.markdown("""
        **Visualize agent movement and state evolution**
        - Select which state dimensions to plot
        - Filter by time range
        - Color by time or other metrics
        """)
        
        col1, col2 = st.columns([2, 1])
        
        with col2:
            st.subheader("⚙️ Controls")
            
            # Dimension selection
            available_dims = list(range(states.shape[1]))
            selected_dims = st.multiselect(
                "Select dimensions to plot",
                available_dims,
                default=available_dims[:2] if len(available_dims) >= 2 else available_dims
            )
            
            # Time range filter
            max_time = len(states)
            time_range = st.slider(
                "Time range",
                0, max_time, (0, min(1000, max_time))
            )
            
            color_by_time = st.checkbox("Color by time", value=True)
            
            # Statistics
            st.metric("Selected Range", f"{time_range[1] - time_range[0]} steps")
            if len(selected_dims) >= 1:
                dim_mean = np.mean(states[time_range[0]:time_range[1], selected_dims[0]])
                st.metric(f"Mean (Dim {selected_dims[0]})", f"{dim_mean:.3f}")
        
        with col1:
            # Filter data
            states_filtered = states[time_range[0]:time_range[1]]
            
            if len(selected_dims) >= 2:
                fig_traj = plot_trajectory_interactive(states_filtered, selected_dims, color_by_time=color_by_time)
                st.plotly_chart(fig_traj, use_container_width=True)
            else:
                st.warning("Please select at least 2 dimensions to plot trajectory.")
            
            # Time series plot
            if selected_dims:
                st.subheader("State Time Series")
                fig_ts = go.Figure()
                for dim in selected_dims[:5]:  # Limit to 5 dimensions
                    fig_ts.add_trace(go.Scatter(
                        x=np.arange(time_range[0], time_range[1]),
                        y=states_filtered[:, dim],
                        mode='lines',
                        name=f'Dim {dim}'
                    ))
                fig_ts.update_layout(xaxis_title='Timestep', yaxis_title='Value', height=400)
                st.plotly_chart(fig_ts, use_container_width=True)
        
        # Raw Data Table
        st.markdown("---")
        st.subheader("📋 Raw Data Table")
        
        col_display, col_export = st.columns([3, 1])
        
        with col_display:
            show_rows = st.slider("Number of rows to display", 10, 100, 50, 10, key="traj_rows")
        
        with col_export:
            st.write("")  # Spacer
            st.write("")  # Spacer
            
            # Prepare CSV export
            df_export = pd.DataFrame(
                np.hstack([
                    np.arange(len(states)).reshape(-1, 1),
                    states,
                    actions,
                    rewards.reshape(-1, 1)
                ]),
                columns=['timestep'] + 
                        [f'state_{i}' for i in range(states.shape[1])] +
                        [f'action_{i}' for i in range(actions.shape[1])] +
                        ['reward']
            )
            
            csv = df_export.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name=f"{st.session_state.current_dataset}_data.csv",
                mime="text/csv"
            )
        
        # Display table
        df_display = pd.DataFrame(
            np.hstack([
                np.arange(len(states[:show_rows])).reshape(-1, 1),
                states[:show_rows],
                actions[:show_rows],
                rewards[:show_rows].reshape(-1, 1)
            ]),
            columns=['timestep'] + 
                    [f'state_{i}' for i in range(states.shape[1])] +
                    [f'action_{i}' for i in range(actions.shape[1])] +
                    ['reward']
        )
        
        st.dataframe(
            df_display.style.format("{:.4f}"),
            use_container_width=True,
            height=400
        )
    
    # Rewards & Actions
    with tabs[tab_offset + 1]:
        st.header("Rewards & Actions Analysis")
        st.markdown("""
        **Analyze learning progress and agent behavior**
        - View reward trends with customizable smoothing
        - Examine action patterns
        - Calculate performance metrics
        """)
        
        col1, col2 = st.columns([2, 1])
        
        with col2:
            st.subheader("⚙️ Controls")
            
            window_size = st.slider("Smoothing window", 1, 100, 10, key="reward_window")
            
            st.subheader("Display Options")
            show_raw = st.checkbox("Show raw rewards", value=True)
            show_smoothed = st.checkbox("Show smoothed rewards", value=True)
            show_cumulative = st.checkbox("Show cumulative rewards", value=True)
            
            # Metrics
            st.subheader("📈 Metrics")
            st.metric("Mean Reward", f"{np.mean(rewards):.3f}")
            st.metric("Max Reward", f"{np.max(rewards):.3f}")
            st.metric("Total Reward", f"{np.sum(rewards):.3f}")
            st.metric("Std Dev", f"{np.std(rewards):.3f}")
        
        with col1:
            # Reward plot
            fig_reward = plot_rewards_interactive(rewards, window_size, show_raw, show_smoothed, show_cumulative)
            st.plotly_chart(fig_reward, use_container_width=True)
            
            # Action plot
            st.subheader("Actions Over Time")
            fig_action = go.Figure()
            for i in range(actions.shape[1]):
                fig_action.add_trace(go.Scatter(
                    x=np.arange(len(actions)),
                    y=actions[:, i],
                    mode='lines',
                    name=f'Action {i}'
                ))
            fig_action.update_layout(xaxis_title='Timestep', yaxis_title='Action Value', height=400)
            st.plotly_chart(fig_action, use_container_width=True)
            
            # Reward-Action correlation
            st.subheader("Reward vs Actions")
            fig_reward_action = go.Figure()
            for i in range(actions.shape[1]):
                fig_reward_action.add_trace(go.Scatter(
                    x=actions[:, i],
                    y=rewards,
                    mode='markers',
                    name=f'Action {i}',
                    opacity=0.6
                ))
            fig_reward_action.update_layout(
                xaxis_title='Action Value',
                yaxis_title='Reward',
                height=400
            )
            st.plotly_chart(fig_reward_action, use_container_width=True)
        
        # Raw Data Table
        st.markdown("---")
        st.subheader("📋 Rewards & Actions Data")
        
        show_reward_rows = st.slider("Number of rows to display", 10, 100, 50, 10, key="reward_rows")
        
        # Display table
        df_rewards = pd.DataFrame(
            np.hstack([
                np.arange(len(rewards[:show_reward_rows])).reshape(-1, 1),
                actions[:show_reward_rows],
                rewards[:show_reward_rows].reshape(-1, 1)
            ]),
            columns=['timestep'] + 
                    [f'action_{i}' for i in range(actions.shape[1])] +
                    ['reward']
        )
        
        st.dataframe(
            df_rewards.style.format("{:.4f}"),
            use_container_width=True,
            height=400
        )
    
    # Symmetry Analysis
    with tabs[tab_offset + 2]:
        st.header("Symmetry Analysis")
        st.markdown("""
        **Detect and analyze symmetry in agent behavior**
        - Test multiple geometric transformations
        - Adjust detection threshold
        - Visualize transformed trajectories
        """)
        
        col1, col2 = st.columns([2, 1])
        
        with col2:
            st.subheader("⚙️ Controls")
            
            # Create a temporary detector to get available transformations
            temp_detector = SymmetryDetector(state_dims=states.shape[1], action_dims=actions.shape[1])
            available_transforms = list(temp_detector.transformations.keys())
            
            selected_transforms = st.multiselect(
                "Select transformations to test",
                available_transforms,
                default=available_transforms[:4]
            )
            
            threshold = st.slider("Symmetry threshold", 0.0, 1.0, 0.7, 0.05)
            
            if st.button("🔍 Analyze Symmetry"):
                with st.spinner("Analyzing symmetry..."):
                    symmetry_scores, transformed_data = analyze_symmetry_interactive(
                        states, actions, rewards, selected_transforms, threshold
                    )
                    st.session_state.symmetry_results[st.session_state.current_dataset] = {
                        "scores": symmetry_scores,
                        "transformed_data": transformed_data
                    }
                st.success("✅ Analysis complete!")
        
        with col1:
            if st.session_state.current_dataset in st.session_state.symmetry_results:
                results = st.session_state.symmetry_results[st.session_state.current_dataset]
                scores = results["scores"]
                transformed_data = results["transformed_data"]
                
                # Scores bar chart
                st.subheader("Symmetry Scores")
                sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                trans_names = [item[0] for item in sorted_items]
                score_values = [item[1] for item in sorted_items]
                
                fig_sym = px.bar(
                    x=trans_names,
                    y=score_values,
                    labels={"x": "Transformation", "y": "Score"},
                    color=score_values,
                    color_continuous_scale="Viridis"
                )
                fig_sym.add_hline(y=threshold, line_dash="dash", line_color="red", annotation_text="Threshold")
                fig_sym.update_layout(yaxis_range=[0, 1], height=400)
                st.plotly_chart(fig_sym, use_container_width=True)
                
                # Transformation visualization
                st.subheader("Transformation Comparison")
                selected_trans = st.selectbox(
                    "Select transformation to visualize",
                    list(scores.keys()),
                    format_func=lambda x: f"{x} (score: {scores[x]:.3f})"
                )
                
                if selected_trans and states.shape[1] >= 2:
                    trans_states = transformed_data[selected_trans]["states"]
                    
                    fig_comp = make_subplots(rows=1, cols=2, subplot_titles=["Original", f"{selected_trans}"])
                    
                    fig_comp.add_trace(
                        go.Scatter(x=states[:500, 0], y=states[:500, 1], mode='lines', name='Original'),
                        row=1, col=1
                    )
                    fig_comp.add_trace(
                        go.Scatter(x=trans_states[:500, 0], y=trans_states[:500, 1], mode='lines', name='Transformed'),
                        row=1, col=2
                    )
                    
                    fig_comp.update_layout(height=400, showlegend=False)
                    st.plotly_chart(fig_comp, use_container_width=True)
                
                # Download results
                if st.button("💾 Download Results"):
                    json_str = json.dumps(scores, indent=2)
                    st.download_button(
                        label="Download JSON",
                        data=json_str,
                        file_name=f"symmetry_scores_{st.session_state.current_dataset}.json",
                        mime="application/json"
                    )
            else:
                st.info("Click 'Analyze Symmetry' to start analysis.")
    
    # Clustering
    with tabs[tab_offset + 3]:
        st.header("Behavior Clustering & Anomaly Detection")
        st.markdown("""
        **Group similar behaviors and detect outliers**
        - Choose clustering algorithm (K-Means, DBSCAN)
        - Adjust number of clusters
        - Identify anomalous behaviors
        """)
        
        col1, col2 = st.columns([2, 1])
        
        with col2:
            st.subheader("⚙️ Controls")
            
            algorithm = st.selectbox("Clustering Algorithm", ["K-Means", "DBSCAN"])
            
            if algorithm == "K-Means":
                n_clusters = st.slider("Number of clusters", 2, 10, 3)
            else:
                n_clusters = 3
            
            n_components = st.slider("PCA components", 2, min(10, states.shape[1]), 2)
            
            anomaly_percentile = st.slider("Anomaly threshold (percentile)", 90, 99, 95)
            
            if st.button("🧩 Run Clustering"):
                with st.spinner("Clustering behaviors..."):
                    X_pca, labels, centers, variance_ratio = cluster_behaviors_interactive(
                        states, actions, n_clusters, algorithm, n_components
                    )
                    
                    st.session_state.analysis_cache[f"{st.session_state.current_dataset}_cluster"] = {
                        "X_pca": X_pca,
                        "labels": labels,
                        "centers": centers,
                        "variance_ratio": variance_ratio,
                        "algorithm": algorithm,
                        "n_clusters": n_clusters
                    }
                st.success("✅ Clustering complete!")
            
            # PCA info
            cache_key = f"{st.session_state.current_dataset}_cluster"
            if cache_key in st.session_state.analysis_cache:
                variance_ratio = st.session_state.analysis_cache[cache_key]["variance_ratio"]
                st.subheader("📊 PCA Info")
                st.metric("Variance Explained", f"{sum(variance_ratio)*100:.1f}%")
        
        with col1:
            cache_key = f"{st.session_state.current_dataset}_cluster"
            if cache_key in st.session_state.analysis_cache:
                cache = st.session_state.analysis_cache[cache_key]
                X_pca = cache["X_pca"]
                labels = cache["labels"]
                centers = cache["centers"]
                
                # Cluster plot
                st.subheader("Behavior Clusters")
                fig_cluster = px.scatter(
                    x=X_pca[:, 0], y=X_pca[:, 1],
                    color=labels.astype(str),
                    labels={"x": "PC1", "y": "PC2", "color": "Cluster"},
                    title=f"{cache['algorithm']} Clustering"
                )
                
                if centers is not None:
                    fig_cluster.add_trace(go.Scatter(
                        x=centers[:, 0], y=centers[:, 1],
                        mode='markers',
                        marker=dict(size=15, color='red', symbol='x'),
                        name='Centers'
                    ))
                
                fig_cluster.update_layout(height=500)
                st.plotly_chart(fig_cluster, use_container_width=True)
                
                # Anomaly detection
                st.subheader("Anomaly Detection")
                if centers is not None:
                    dists = np.min(cdist(X_pca, centers), axis=1)
                else:
                    # For DBSCAN, use distance to nearest neighbors
                    dists = np.linalg.norm(X_pca - X_pca.mean(axis=0), axis=1)
                
                threshold_val = np.percentile(dists, anomaly_percentile)
                anomalies = dists > threshold_val
                n_anomalies = np.sum(anomalies)
                
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("Anomalies Detected", n_anomalies)
                with col_b:
                    st.metric("Anomaly Rate", f"{n_anomalies/len(dists)*100:.1f}%")
                
                # Anomaly plot
                fig_anom = px.scatter(
                    x=X_pca[:, 0], y=X_pca[:, 1],
                    color=anomalies,
                    labels={"x": "PC1", "y": "PC2", "color": "Anomaly"},
                    title="Anomaly Detection"
                )
                fig_anom.update_layout(height=400)
                st.plotly_chart(fig_anom, use_container_width=True)
            else:
                st.info("Click 'Run Clustering' to start analysis.")
    
    # Advanced Analysis
    with tabs[tab_offset + 4]:
        st.header("Advanced Analysis Tools")
        st.markdown("""
        **Deep dive into agent behavior with advanced visualizations**
        - State-action correlations
        - 3D trajectory visualization
        - Velocity and acceleration analysis
        - Action energy metrics
        - Statistical reports
        """)
        
        analysis_tabs = st.tabs(["📊 Correlations", "🌐 3D View", "⚡ Dynamics", "🔋 Energy", "📈 Statistics"])
        
        # Correlations
        with analysis_tabs[0]:
            st.subheader("State-Action Correlation Analysis")
            st.markdown("""
            **Understanding how states influence actions**
            - Red: Positive correlation (state increases → action increases)
            - Blue: Negative correlation (state increases → action decreases)
            - White: No correlation
            """)
            
            fig_corr = plot_state_action_correlation(states, actions)
            st.plotly_chart(fig_corr, use_container_width=True)
            
            # Additional correlation insights
            st.subheader("Correlation Insights")
            combined = np.hstack([states, actions])
            full_corr = np.corrcoef(combined.T)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Avg State Correlation", f"{np.mean(np.abs(full_corr[:states.shape[1], :states.shape[1]])):.3f}")
            with col2:
                st.metric("Avg Action Correlation", f"{np.mean(np.abs(full_corr[states.shape[1]:, states.shape[1]:])):.3f}")
        
        # 3D Visualization
        with analysis_tabs[1]:
            st.subheader("3D Trajectory Visualization")
            st.markdown("""
            **Explore agent movement in 3D space**
            - First 3 dimensions of state space
            - Color gradient shows temporal progression
            - Rotate and zoom to explore from different angles
            """)
            
            if states.shape[1] >= 3:
                color_3d = st.checkbox("Color by time (3D)", value=True)
                
                # Time range for 3D
                max_3d = min(1000, len(states))
                range_3d = st.slider("3D visualization range", 0, len(states), (0, max_3d), key="3d_range")
                
                states_3d = states[range_3d[0]:range_3d[1]]
                fig_3d = plot_trajectory_3d(states_3d, color_by_time=color_3d)
                
                if fig_3d:
                    st.plotly_chart(fig_3d, use_container_width=True)
                    
                    # 3D Statistics
                    st.subheader("3D Trajectory Metrics")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        path_length = np.sum(np.linalg.norm(np.diff(states_3d[:, :3], axis=0), axis=1))
                        st.metric("Path Length", f"{path_length:.2f}")
                    with col2:
                        bbox_size = np.ptp(states_3d[:, :3], axis=0)
                        st.metric("Bounding Box Volume", f"{np.prod(bbox_size):.3f}")
                    with col3:
                        centroid = np.mean(states_3d[:, :3], axis=0)
                        st.metric("Centroid Distance", f"{np.linalg.norm(centroid):.3f}")
            else:
                st.warning("Need at least 3 state dimensions for 3D visualization")
        
        # Dynamics Analysis
        with analysis_tabs[2]:
            st.subheader("Velocity & Acceleration Analysis")
            st.markdown("""
            **Analyze movement dynamics**
            - Velocity: Rate of state change (first derivative)
            - Acceleration: Rate of velocity change (second derivative)
            - High values indicate rapid movements or jerky behavior
            """)
            
            fig_dynamics, velocity_mag, accel_mag = plot_velocity_acceleration(states)
            st.plotly_chart(fig_dynamics, use_container_width=True)
            
            # Dynamics statistics
            st.subheader("Dynamics Metrics")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Mean Velocity", f"{np.mean(velocity_mag):.4f}")
            with col2:
                st.metric("Max Velocity", f"{np.max(velocity_mag):.4f}")
            with col3:
                st.metric("Mean Acceleration", f"{np.mean(accel_mag):.4f}")
            with col4:
                st.metric("Max Acceleration", f"{np.max(accel_mag):.4f}")
            
            # Smoothness analysis
            st.subheader("Movement Smoothness")
            jerk = np.diff(accel_mag)  # Third derivative
            smoothness_score = 1.0 / (1.0 + np.std(jerk))
            st.metric("Smoothness Score", f"{smoothness_score:.3f}", 
                     help="Higher = smoother movement (0-1 scale)")
        
        # Energy Analysis
        with analysis_tabs[3]:
            st.subheader("Action Energy Analysis")
            st.markdown("""
            **Measure control effort and efficiency**
            - Action Energy: Magnitude of action vectors
            - High energy = aggressive control
            - Low energy = gentle/efficient control
            """)
            
            fig_energy, action_energy = plot_action_energy(actions)
            st.plotly_chart(fig_energy, use_container_width=True)
            
            # Energy statistics
            st.subheader("Energy Metrics")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Energy", f"{np.sum(action_energy):.2f}")
            with col2:
                st.metric("Mean Energy", f"{np.mean(action_energy):.4f}")
            with col3:
                st.metric("Peak Energy", f"{np.max(action_energy):.4f}")
            with col4:
                efficiency = np.sum(rewards) / (np.sum(action_energy) + 1e-6)
                st.metric("Efficiency (Reward/Energy)", f"{efficiency:.3f}")
            
            # Energy distribution
            st.subheader("Energy Distribution")
            fig_hist = px.histogram(x=action_energy, nbins=50, 
                                   labels={"x": "Action Energy", "y": "Frequency"},
                                   title="Action Energy Distribution")
            fig_hist.update_layout(height=300)
            st.plotly_chart(fig_hist, use_container_width=True)
        
        # Statistics Report
        with analysis_tabs[4]:
            st.subheader("Comprehensive Statistics Report")
            st.markdown("""
            **Detailed statistical summary of all data**
            - Mean, standard deviation, min/max for all dimensions
            - Downloadable as JSON
            """)
            
            stats = generate_statistics_report(states, actions, rewards)
            
            # Rewards stats
            st.subheader("📊 Reward Statistics")
            reward_df = pd.DataFrame([stats["Rewards"]]).T
            reward_df.columns = ["Value"]
            st.dataframe(reward_df, use_container_width=True)
            
            # States stats
            st.subheader("📐 State Statistics")
            state_stats_dict = {}
            for stat_name, values in stats["States"].items():
                for i, val in enumerate(values):
                    if f"Dim {i}" not in state_stats_dict:
                        state_stats_dict[f"Dim {i}"] = {}
                    state_stats_dict[f"Dim {i}"][stat_name] = f"{val:.4f}"
            
            state_df = pd.DataFrame(state_stats_dict).T
            st.dataframe(state_df, use_container_width=True)
            
            # Actions stats
            st.subheader("🎮 Action Statistics")
            action_stats_dict = {}
            for stat_name, values in stats["Actions"].items():
                for i, val in enumerate(values):
                    if f"Action {i}" not in action_stats_dict:
                        action_stats_dict[f"Action {i}"] = {}
                    action_stats_dict[f"Action {i}"][stat_name] = f"{val:.4f}"
            
            action_df = pd.DataFrame(action_stats_dict).T
            st.dataframe(action_df, use_container_width=True)
            
            # Download button
            if st.button("💾 Download Full Report (JSON)"):
                # Convert numpy arrays to lists for JSON serialization
                json_stats = {}
                for key, val in stats.items():
                    json_stats[key] = {}
                    for sub_key, sub_val in val.items():
                        if isinstance(sub_val, np.ndarray):
                            json_stats[key][sub_key] = sub_val.tolist()
                        else:
                            json_stats[key][sub_key] = float(sub_val)
                
                json_str = json.dumps(json_stats, indent=2)
                st.download_button(
                    label="Download JSON",
                    data=json_str,
                    file_name=f"statistics_report_{st.session_state.current_dataset}.json",
                    mime="application/json"
                )
    
    # About
    with tabs[-1]:
        st.header("About This Dashboard")
        
        st.markdown("""
        ## 🎯 Symmetry-Aware Reinforcement Learning Dashboard
        
        ### Project Goals
        - **Visualize and interpret symmetry** in RL trajectories
        - **Build dashboards** for reward and performance analysis
        - **Prototype guided RL pipelines** inspired by geometric frameworks
        - **Automate RL experiments** and log parsing
        - **Evaluate agent behavior** through clustering and anomaly detection
        
        ### Key Features
        
        #### 🎯 Trajectory Analysis
        - Interactive dimension selection
        - Time range filtering
        - Real-time state visualization
        
        #### 💰 Rewards & Actions
        - Customizable smoothing
        - Multiple display modes
        - Performance metrics
        
        #### 🔄 Symmetry Detection
        - Multiple transformation types (mirror, rotation, time-reversal)
        - Adjustable thresholds
        - Visual comparison of transformed trajectories
        
        #### 🧩 Behavior Clustering
        - Multiple algorithms (K-Means, DBSCAN)
        - PCA dimensionality reduction
        - Anomaly detection
        
        #### 📊 Multi-Dataset Comparison
        - Load multiple experiments
        - Side-by-side comparisons
        - Summary statistics
        
        ### Research Context
        
        This dashboard supports research in **symmetry-aware reinforcement learning for robotic locomotion**.
        
        **Key References:**
        - [Hu & Dear, 2023] *Detecting and Exploiting Symmetry to Accelerate Reinforcement Learning*
        - [Pujari et al., 2023] *Exploiting Symmetry to Accelerate Reinforcement Learning*
        
        ### About Demo Data
        
        The demo data is a **synthetic dataset** designed to showcase dashboard features:
        - Contains **1000 timesteps** with configurable parameters
        - **Periodic patterns** mimicking natural locomotion
        - **Built-in symmetry** for easy testing (second half mirrors first half)
        - **Adjustable noise** to simulate real-world conditions
        - **Not from actual experiments** - for demonstration only
        
        Use demo data to:
        - Learn how the dashboard works
        - Test symmetry detection algorithms
        - Understand visualization options
        - Before uploading real RL logs
        
        ### How to Use
        
        1. **Load Data**: Upload CSV or generate demo data from sidebar
        2. **Explore Trajectories**: Select dimensions and time ranges
        3. **Analyze Rewards**: Adjust smoothing and view metrics
        4. **Detect Symmetry**: Choose transformations and run analysis
        5. **Cluster Behaviors**: Select algorithm and parameters
        6. **Compare Experiments**: Load multiple datasets for comparison
        7. **Download Results**: Save analysis results as JSON
        
        ### Contact & Contribution
        
        This project is part of ongoing research in robotics and reinforcement learning.
        For questions or collaboration opportunities, please refer to the GitHub repository.
        """)
        
        # Session info
        with st.expander("🔧 Session Information"):
            st.write("Loaded Datasets:", len(st.session_state.datasets))
            st.write("Current Dataset:", st.session_state.current_dataset)
            st.write("Comparison Mode:", st.session_state.comparison_mode)
            st.write("Cached Analyses:", len(st.session_state.analysis_cache))


if __name__ == "__main__":
    main()


