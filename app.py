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
    """Initialize session state variables."""
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


def create_demo_data(timesteps=1000, dim_state=8, dim_action=2, noise_level=0.1):
    """
    Create synthetic data with configurable parameters.
    
    Args:
        timesteps: Number of timesteps
        dim_state: State dimension
        dim_action: Action dimension
        noise_level: Amount of random noise (0-1)
    """
    np.random.seed(42)
    
    states = np.zeros((timesteps, dim_state))
    actions = np.zeros((timesteps, dim_action))
    rewards = np.zeros(timesteps)
    next_states = np.zeros((timesteps, dim_state))
    
    for t in range(timesteps // 2):
        # Position with periodic pattern
        states[t, 0] = np.sin(t/50)
        states[t, 1] = np.cos(t/50)
        states[t, 2:4] = noise_level * np.random.normal(0, 0.1, 2)
        states[t, 4:] = 0.1 * noise_level * np.random.normal(0, 1, dim_state-4)
        
        # Actions with periodic pattern
        actions[t, 0] = 0.5 * np.sin(t/30)
        if dim_action > 1:
            actions[t, 1] = 0.5 * np.cos(t/30)
        
        # Mirror for second half
        mirror_t = timesteps - t - 1
        states[mirror_t] = states[t].copy()
        states[mirror_t, 0] *= -1
        if dim_state > 4:
            states[mirror_t, 4:] *= -1
        
        actions[mirror_t] = actions[t].copy()
        actions[mirror_t, 0] *= -1
        
        # Rewards
        rewards[t] = rewards[mirror_t] = np.abs(np.sin(t/100))
        
        if t < timesteps // 2 - 1:
            next_states[t] = states[t+1]
            next_states[mirror_t-1] = states[mirror_t]
    
    next_states[-1] = states[0]
    
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
        st.info("👈 Please load or generate data from the sidebar to begin analysis.")
        
        # Show demo information
        with st.expander("📖 About Demo Data"):
            st.markdown("""
            **Demo Data Characteristics:**
            
            - **Synthetic dataset** created for demonstration purposes
            - Contains **periodic and symmetric patterns** mimicking robot locomotion
            - First half follows smooth sinusoidal patterns
            - Second half is a **mirrored version** to demonstrate symmetry detection
            - Includes configurable noise to simulate real-world data
            - **Not from actual robot experiments** - designed for testing dashboard features
            
            **How to use:**
            1. Adjust parameters in the sidebar (timesteps, dimensions, noise level)
            2. Click "Generate Data" to create synthetic trajectories
            3. Explore all dashboard features with this data
            4. Upload your own RL logs when ready for real analysis
            """)
        
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
        tabs = st.tabs(["📊 Comparison", "🎯 Trajectory", "💰 Rewards", "🔄 Symmetry", "🧩 Clustering", "ℹ️ About"])
        
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
        tabs = st.tabs(["🎯 Trajectory", "💰 Rewards", "🔄 Symmetry", "🧩 Clustering", "ℹ️ About"])
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


