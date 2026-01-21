"""
Streamlit App for RL Symmetry Visualization

This application provides an interactive dashboard for visualizing 
reinforcement learning trajectories and analyzing symmetry patterns.
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

# Add the project root to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.symmetry_detector import SymmetryDetector, load_trajectory_data
from scripts.analyze_logs import compute_moving_average


def init_session_state():
    """Initialize session state variables."""
    if 'loaded_data' not in st.session_state:
        st.session_state.loaded_data = None
    if 'symmetry_scores' not in st.session_state:
        st.session_state.symmetry_scores = None
    if 'transformed_data' not in st.session_state:
        st.session_state.transformed_data = {}


def plot_trajectory_2d(states, title="Agent Trajectory"):
    """
    Plot 2D trajectory visualization.
    
    Args:
        states: Array of states with at least 2 dimensions
        title: Plot title
    """
    if states.shape[1] < 2:
        st.warning("Cannot create 2D trajectory plot: state has fewer than 2 dimensions")
        return None
    
    fig = px.line(x=states[:, 0], y=states[:, 1], 
                  title=title,
                  labels={"x": "Dimension 0", "y": "Dimension 1"})
    
    fig.update_layout(
        hovermode="closest",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    
    return fig


def plot_state_time_series(states, timesteps=None, title="State Dimensions Over Time"):
    """
    Plot state dimensions as time series.
    
    Args:
        states: Array of states
        timesteps: Array of timestep values (if None, uses range)
        title: Plot title
    """
    if timesteps is None:
        timesteps = np.arange(len(states))
    
    fig = go.Figure()
    
    for i in range(min(states.shape[1], 8)):  # Limit to first 8 dimensions to avoid clutter
        fig.add_trace(go.Scatter(
            x=timesteps,
            y=states[:, i],
            mode='lines',
            name=f'Dim {i}'
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Timestep',
        yaxis_title='Value',
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    
    return fig


def plot_rewards(rewards, timesteps=None, window=10, title="Rewards Over Time"):
    """
    Plot reward time series with smoothing.
    
    Args:
        rewards: Array of reward values
        timesteps: Array of timestep values (if None, uses range)
        window: Window size for moving average
        title: Plot title
    """
    if timesteps is None:
        timesteps = np.arange(len(rewards))
    
    # Calculate smoothed rewards
    if len(rewards) > window:
        smoothed_rewards = compute_moving_average(rewards, window)
        smoothed_timesteps = timesteps[window-1:]
    else:
        smoothed_rewards = rewards
        smoothed_timesteps = timesteps
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add raw rewards
    fig.add_trace(
        go.Scatter(
            x=timesteps,
            y=rewards,
            mode='lines',
            name='Raw Rewards',
            line=dict(color='rgba(0,0,255,0.3)'),
            showlegend=True
        )
    )
    
    # Add smoothed rewards
    fig.add_trace(
        go.Scatter(
            x=smoothed_timesteps,
            y=smoothed_rewards,
            mode='lines',
            name=f'Smoothed (window={window})',
            line=dict(color='blue', width=2),
            showlegend=True
        )
    )
    
    # Add cumulative rewards on secondary axis
    fig.add_trace(
        go.Scatter(
            x=timesteps,
            y=np.cumsum(rewards),
            mode='lines',
            name='Cumulative Reward',
            line=dict(color='green'),
            showlegend=True
        ),
        secondary_y=True
    )
    
    fig.update_layout(
        title=title,
        xaxis_title='Timestep',
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    
    fig.update_yaxes(title_text="Reward", secondary_y=False)
    fig.update_yaxes(title_text="Cumulative Reward", secondary_y=True)
    
    return fig


def plot_actions(actions, timesteps=None, title="Actions Over Time"):
    """
    Plot action dimensions as time series.
    
    Args:
        actions: Array of actions
        timesteps: Array of timestep values (if None, uses range)
        title: Plot title
    """
    if timesteps is None:
        timesteps = np.arange(len(actions))
    
    fig = go.Figure()
    
    for i in range(actions.shape[1]):
        fig.add_trace(go.Scatter(
            x=timesteps,
            y=actions[:, i],
            mode='lines',
            name=f'Action {i}'
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Timestep',
        yaxis_title='Action Value',
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    
    return fig


def plot_symmetry_scores(symmetry_scores, title="Symmetry Scores"):
    """
    Plot symmetry scores as bar chart.
    
    Args:
        symmetry_scores: Dictionary mapping transformation names to scores
        title: Plot title
    """
    # Sort transformations by score
    sorted_items = sorted(symmetry_scores.items(), key=lambda x: x[1], reverse=True)
    trans_names = [item[0] for item in sorted_items]
    scores = [item[1] for item in sorted_items]
    
    fig = px.bar(
        x=trans_names, 
        y=scores,
        title=title,
        labels={"x": "Transformation", "y": "Symmetry Score"},
        text=scores,
        color=scores,
        color_continuous_scale="Viridis"
    )
    
    fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
    
    fig.update_layout(
        xaxis_title="",
        yaxis_title="Symmetry Score",
        yaxis_range=[0, 1],
        coloraxis_showscale=False,
        hovermode="x unified",
    )
    
    return fig


def compare_transformed_trajectory(original_states, transformed_states, transformation_name):
    """
    Plot original vs transformed trajectory.
    
    Args:
        original_states: Original state trajectory
        transformed_states: Transformed state trajectory
        transformation_name: Name of the transformation
    """
    if original_states.shape[1] < 2 or transformed_states.shape[1] < 2:
        st.warning("Cannot create trajectory comparison: states have fewer than 2 dimensions")
        return None
    
    fig = make_subplots(rows=1, cols=2, 
                        subplot_titles=["Original Trajectory", f"After {transformation_name}"])
    
    # Original trajectory
    fig.add_trace(
        go.Scatter(
            x=original_states[:, 0],
            y=original_states[:, 1],
            mode='lines',
            name='Original',
            line=dict(color='blue'),
        ),
        row=1, col=1
    )
    
    # Transformed trajectory
    fig.add_trace(
        go.Scatter(
            x=transformed_states[:, 0],
            y=transformed_states[:, 1],
            mode='lines',
            name='Transformed',
            line=dict(color='red'),
        ),
        row=1, col=2
    )
    
    fig.update_layout(
        title=f"Trajectory Transformation: {transformation_name}",
        hovermode="closest",
    )
    
    fig.update_xaxes(title_text="Dimension 0", row=1, col=1)
    fig.update_xaxes(title_text="Dimension 0", row=1, col=2)
    fig.update_yaxes(title_text="Dimension 1", row=1, col=1)
    fig.update_yaxes(title_text="Dimension 1", row=1, col=2)
    
    return fig


def detect_and_analyze_symmetry(states, actions, rewards, next_states=None):
    """
    Detect symmetry in trajectory data.
    
    Args:
        states: Array of states
        actions: Array of actions
        rewards: Array of rewards
        next_states: Optional array of next states
        
    Returns:
        Dictionary of symmetry scores
    """
    # Initialize detector
    detector = SymmetryDetector(state_dims=states.shape[1], action_dims=actions.shape[1])
    
    # Detect symmetry
    symmetry_scores = detector.detect_symmetry(states, actions, rewards, next_states)
    
    # Store transformations for visualization
    for trans_name in symmetry_scores.keys():
        transform_fn = detector.transformations[trans_name]["transform_fn"]
        
        # Apply transformation
        transformed_states = np.zeros_like(states)
        transformed_actions = np.zeros_like(actions)
        
        for t in range(len(states)):
            transformed_states[t], transformed_actions[t] = transform_fn(states[t], actions[t])
        
        # Store transformed data
        st.session_state.transformed_data[trans_name] = {
            "states": transformed_states,
            "actions": transformed_actions
        }
    
    return symmetry_scores


def create_sample_data():
    """
    Create synthetic data for demonstration.
    
    Returns:
        Tuple of (states, actions, rewards, next_states)
    """
    # Create a synthetic trajectory with built-in symmetry
    np.random.seed(42)
    timesteps = 1000
    dim_state = 8
    dim_action = 2
    
    # Create sample state trajectory with time symmetry (second half mirrors first half)
    states = np.zeros((timesteps, dim_state))
    actions = np.zeros((timesteps, dim_action))
    rewards = np.zeros(timesteps)
    next_states = np.zeros((timesteps, dim_state))
    
    for t in range(timesteps // 2):
        # Generate a sinusoidal pattern for part of the state
        states[t, 0] = np.sin(t/50)  # x position
        states[t, 1] = np.cos(t/50)  # y position
        states[t, 2:4] = np.random.normal(0, 0.1, 2)  # random joint angles
        states[t, 4:] = 0.1 * np.random.normal(0, 1, 4)  # velocities
        
        # Actions also follow a pattern
        actions[t, 0] = 0.5 * np.sin(t/30)
        actions[t, 1] = 0.5 * np.cos(t/30)
        
        # Mirror for second half to create time symmetry
        mirror_t = timesteps - t - 1
        states[mirror_t] = states[t].copy()
        states[mirror_t, 0] *= -1  # Flip x position for mirror symmetry
        states[mirror_t, 4:] *= -1  # Flip velocities
        
        actions[mirror_t] = actions[t].copy()
        actions[mirror_t, 0] *= -1  # Flip first action
        
        # Rewards are symmetric
        rewards[t] = rewards[mirror_t] = np.abs(np.sin(t/100))
        
        # Create next states (just shifted by 1)
        if t < timesteps // 2 - 1:
            next_states[t] = states[t+1]
            next_states[mirror_t-1] = states[mirror_t]
    
    next_states[-1] = states[0]  # Close the loop
    
    return states, actions, rewards, next_states


def main():
    """Main function for the Streamlit app."""
    st.set_page_config(
        page_title="RL Symmetry Visualization",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    
    # Initialize session state
    init_session_state()
    
    # Sidebar
    st.sidebar.title("🤖 RL Symmetry Visualization")
    
    # Data loading options
    st.sidebar.header("Data")
    data_source = st.sidebar.radio(
        "Select data source",
        options=["Upload CSV", "Use Demo Data"]
    )
    
    if data_source == "Upload CSV":
        uploaded_file = st.sidebar.file_uploader(
            "Upload trajectory data (CSV)",
            type=["csv"],
            help="CSV should contain columns prefixed with 'state_', 'action_', and a 'reward' column"
        )
        
        if uploaded_file is not None:
            try:
                # Save uploaded file to temp location
                upload_dir = Path("temp_uploads")
                upload_dir.mkdir(exist_ok=True)
                
                temp_file = upload_dir / f"upload_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                with open(temp_file, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Load data
                states, actions, rewards, next_states = load_trajectory_data(str(temp_file))
                st.session_state.loaded_data = {
                    "states": states,
                    "actions": actions,
                    "rewards": rewards,
                    "next_states": next_states,
                    "source": str(temp_file)
                }
                st.sidebar.success(f"Data loaded successfully: {states.shape[0]} timesteps")
            except Exception as e:
                st.sidebar.error(f"Error loading data: {str(e)}")
    
    elif data_source == "Use Demo Data":
        if st.sidebar.button("Generate Demo Data"):
            # Create synthetic data
            states, actions, rewards, next_states = create_sample_data()
            st.session_state.loaded_data = {
                "states": states,
                "actions": actions,
                "rewards": rewards,
                "next_states": next_states,
                "source": "demo"
            }
            st.sidebar.success(f"Demo data generated: {states.shape[0]} timesteps")
    
    # Symmetry detection options
    if st.session_state.loaded_data is not None:
        st.sidebar.header("Symmetry Detection")
        
        if st.sidebar.button("Detect Symmetry"):
            states = st.session_state.loaded_data["states"]
            actions = st.session_state.loaded_data["actions"]
            rewards = st.session_state.loaded_data["rewards"]
            next_states = st.session_state.loaded_data["next_states"]
            
            # Detect symmetry
            with st.sidebar:
                with st.spinner("Detecting symmetry..."):
                    symmetry_scores = detect_and_analyze_symmetry(
                        states, actions, rewards, next_states)
                    st.session_state.symmetry_scores = symmetry_scores
            
            st.sidebar.success("Symmetry detection complete!")
        
        # Visualization options
        st.sidebar.header("Visualization Options")
        
        smoothing_window = st.sidebar.slider(
            "Smoothing window size",
            min_value=1,
            max_value=50,
            value=10,
            help="Window size for moving average when plotting rewards"
        )
        
        max_timesteps = st.sidebar.number_input(
            "Max timesteps to plot",
            min_value=10,
            max_value=st.session_state.loaded_data["states"].shape[0],
            value=min(1000, st.session_state.loaded_data["states"].shape[0]),
            help="Limit the number of timesteps to plot for performance"
        )
    
    # Main content
    st.title("Reinforcement Learning Symmetry Visualization")

    # --- Explanations for each tab ---
    trajectory_explanation = """
#### Trajectory Analysis
- **This panel visualizes how the agent (robot) moves in the environment.**
- **2D Trajectory Plot:** Shows the agent's position (e.g., x, y) over time.
- **State Dimensions Over Time:** Plots how each state variable (joint angles, velocities, etc.) changes during an episode.
- **Data Used:** State information from training logs.
- **Why it matters:** Helps you see if the agent develops periodic or symmetric movement patterns, which are important in locomotion tasks.
"""
    rewards_explanation = """
#### Rewards & Actions
- **This panel shows the rewards the agent receives and the actions it takes.**
- **Rewards Over Time:** Plots the reward at each step or episode (with smoothing and cumulative sum).
- **Actions Over Time:** Plots the values of each action dimension over time.
- **Data Used:** Reward and action information from logs.
- **Why it matters:** Reward trends indicate learning progress; action patterns can reveal if the agent is exploiting symmetry or repeating behaviors.
"""
    symmetry_explanation = """
#### Symmetry Analysis
- **This panel analyzes how symmetric the agent's behavior is.**
- **Symmetry Scores:** For each transformation (mirror, time-reversal, rotation), shows how similar the transformed trajectory is to the original (score 0-1).
- **Transformation Visualization:** Compares the original and transformed trajectories side-by-side.
- **Symmetry Score History:** (Prototype) Tracks how symmetry scores change during training.
- **Data Used:** State, action, and reward data.
- **Why it matters:** High symmetry scores mean the agent's behavior is robust to geometric transformations, which can accelerate learning and improve generalization.
"""
    loss_explanation = """
#### Policy Loss Comparison
- **This panel compares the training loss curves of two policies (e.g., Baseline vs. Guided RL).**
- **Policy Loss Curve:** Shows how the loss decreases as training progresses.
- **Why compare?** Lower/faster-converging loss can indicate more efficient or stable learning.
- **Data Used:** Policy loss logs (synthetic or uploaded).
- **Why it matters:** Helps you see if symmetry-guided RL leads to faster or more stable convergence compared to standard RL.
"""
    cluster_explanation = """
#### Agent Behavior Clustering & Anomaly Detection (Prototype)
- **This panel groups similar agent behaviors and flags unusual (anomalous) trajectories.**
- **Clustering:** Uses PCA and k-means to group similar state/action sequences.
- **Anomaly Detection:** Flags outlier episodes that deviate from normal patterns.
- **Data Used:** State and action data from logs.
- **Why it matters:** Can help identify failure cases, novel behaviors, or opportunities for curriculum learning.
"""
    about_explanation = """
#### About
- **This dashboard supports research in symmetry-aware reinforcement learning for robotic locomotion.**
- **Project Goals:**
    - Visualize and interpret symmetry in RL trajectories
    - Build dashboards for reward and performance analysis
    - Prototype guided RL pipelines inspired by geometric frameworks
    - Automate RL experiments and log parsing
    - Evaluate agent behavior (clustering, anomaly detection)
- **References:**
    - [Hu & Dear, 2023] Detecting and Exploiting Symmetry to Accelerate RL
    - [Pujari et al., 2023] Exploiting Symmetry to Accelerate RL
- **For more details, see the research proposal and README.**
"""
    
    if st.session_state.loaded_data is None:
        st.info("👈 Select a data source from the sidebar to get started")
        
        # Show demo image
        st.markdown("### Sample Visualization")
        st.image("../visualizations/sample_gait_plot.png", 
                caption="Sample gait visualization (if available)",
                use_column_width=True)
    else:
        # Get data
        states = st.session_state.loaded_data["states"][:max_timesteps]
        actions = st.session_state.loaded_data["actions"][:max_timesteps]
        rewards = st.session_state.loaded_data["rewards"][:max_timesteps]
        
        # Create tabs for different visualizations
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "Trajectory Analysis", 
            "Rewards & Actions", 
            "Symmetry Analysis",
            "Policy Loss Comparison",
            "Behavior Clustering",
            "About"
        ])
        
        with tab1:
            st.markdown(trajectory_explanation)
            st.header("Trajectory Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # 2D Trajectory visualization
                trajectory_fig = plot_trajectory_2d(states, title="Agent Trajectory (First 2 Dimensions)")
                st.plotly_chart(trajectory_fig, use_container_width=True)
            
            with col2:
                # State dimensions over time
                state_time_fig = plot_state_time_series(states, title="State Dimensions Over Time")
                st.plotly_chart(state_time_fig, use_container_width=True)
        
        with tab2:
            st.markdown(rewards_explanation)
            st.header("Rewards & Actions")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Rewards visualization
                rewards_fig = plot_rewards(rewards, window=smoothing_window, title="Rewards Over Time")
                st.plotly_chart(rewards_fig, use_container_width=True)
            
            with col2:
                # Actions visualization
                actions_fig = plot_actions(actions, title="Actions Over Time")
                st.plotly_chart(actions_fig, use_container_width=True)
        
        with tab3:
            st.markdown(symmetry_explanation)
            st.header("Symmetry Analysis")
            
            if st.session_state.symmetry_scores is None:
                st.info("👈 Click 'Detect Symmetry' in the sidebar to analyze symmetry patterns")
            else:
                col1, col2 = st.columns(2)
                
                with col1:
                    # Symmetry scores visualization
                    scores_fig = plot_symmetry_scores(st.session_state.symmetry_scores, 
                                                     title="Symmetry Scores (higher is more symmetric)")
                    st.plotly_chart(scores_fig, use_container_width=True)
                
                with col2:
                    # Select transformation to visualize
                    selected_trans = st.selectbox(
                        "Select transformation to visualize",
                        options=list(st.session_state.symmetry_scores.keys()),
                        index=0,
                        format_func=lambda x: f"{x} (score: {st.session_state.symmetry_scores[x]:.3f})"
                    )
                    
                    # Transformation visualization
                    trans_fig = compare_transformed_trajectory(
                        states,
                        st.session_state.transformed_data[selected_trans]["states"][:max_timesteps],
                        selected_trans
                    )
                    st.plotly_chart(trans_fig, use_container_width=True)
                
                # Save symmetry results
                if st.button("Save Symmetry Results"):
                    # Create output directory
                    output_dir = Path("../visualizations")
                    output_dir.mkdir(exist_ok=True, parents=True)
                    
                    # Save scores as JSON
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    out_file = output_dir / f"symmetry_scores_{timestamp}.json"
                    
                    with open(out_file, "w") as f:
                        json.dump(st.session_state.symmetry_scores, f, indent=2)
                    
                    st.success(f"Symmetry scores saved to {out_file}")

            # --- Symmetry Score History (Prototype) ---
            st.subheader("Symmetry Score History (Prototype)")
            st.info("This plot shows how symmetry scores change during training. In a real experiment, this would be computed per checkpoint or episode.")
            # Placeholder: generate synthetic symmetry score history
            np.random.seed(42)
            epochs = np.arange(0, 101, 10)
            history = pd.DataFrame({
                'epoch': epochs,
                'mirror_0': np.clip(np.linspace(0.5, 0.92, len(epochs)) + 0.05*np.random.randn(len(epochs)), 0, 1),
                'mirror_1': np.clip(np.linspace(0.3, 0.45, len(epochs)) + 0.05*np.random.randn(len(epochs)), 0, 1),
                'time_reversal': np.clip(np.linspace(0.4, 0.78, len(epochs)) + 0.05*np.random.randn(len(epochs)), 0, 1),
                'rotation_180': np.clip(np.linspace(0.2, 0.31, len(epochs)) + 0.05*np.random.randn(len(epochs)), 0, 1),
            })
            for col in ['mirror_0', 'mirror_1', 'time_reversal', 'rotation_180']:
                st.line_chart(history.set_index('epoch')[col], height=150, use_container_width=True)
        
        with tab4:
            st.markdown(loss_explanation)
            st.header("Policy Loss Comparison")
            st.info("Compare the policy loss curves of two different RL strategies. Upload your own logs or view a synthetic example.")
            # Placeholder: synthetic loss curves
            np.random.seed(42)
            steps = np.arange(0, 100000, 1000)
            loss_baseline = np.exp(-steps/40000) + 0.05*np.random.randn(len(steps))
            loss_guided = np.exp(-steps/25000) + 0.03*np.random.randn(len(steps))
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=steps, y=loss_baseline, mode='lines', name='Baseline Policy'))
            fig.add_trace(go.Scatter(x=steps, y=loss_guided, mode='lines', name='Guided Policy'))
            fig.update_layout(title='Policy Loss Comparison', xaxis_title='Training Steps', yaxis_title='Policy Loss')
            st.plotly_chart(fig, use_container_width=True)
        
        with tab5:
            st.markdown(cluster_explanation)
            st.header("Agent Behavior Clustering & Anomaly Detection (Prototype)")
            st.info("This prototype clusters agent behaviors and flags anomalies using PCA and k-means. Upload your own data for real results.")
            # Placeholder: synthetic clustering
            from sklearn.decomposition import PCA
            from sklearn.cluster import KMeans
            # Use state data for clustering
            X = states[:, :4] if states.shape[1] >= 4 else states
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X)
            kmeans = KMeans(n_clusters=3, random_state=42).fit(X_pca)
            labels = kmeans.labels_
            fig = px.scatter(x=X_pca[:,0], y=X_pca[:,1], color=labels.astype(str),
                             title="Behavior Clusters (PCA projection)", labels={'x':'PC1','y':'PC2','color':'Cluster'})
            st.plotly_chart(fig, use_container_width=True)
            # Anomaly detection (flag points far from cluster centers)
            dists = np.linalg.norm(X_pca - kmeans.cluster_centers_[labels], axis=1)
            anomaly_threshold = np.percentile(dists, 95)
            n_anomalies = np.sum(dists > anomaly_threshold)
            st.write(f"Detected {n_anomalies} anomalous points (top 5% farthest from cluster centers).")
        
        with tab6:
            st.markdown(about_explanation)
            st.header("About")
            
            st.markdown("""
            ## RL-Symmetry-Visualization
            
            This tool helps identify and visualize symmetry patterns in reinforcement learning trajectories.
            
            ### Features:
            
            - **Trajectory Analysis**: Visualize agent movements and state space
            - **Reward Tracking**: Monitor learning progress and performance
            - **Symmetry Detection**: Identify geometric patterns in behavior
            - **Transformation Visualization**: See how symmetry transformations affect trajectories
            
            ### How to Use:
            
            1. Upload a trajectory CSV file or use the demo data
            2. Click "Detect Symmetry" to analyze symmetry patterns
            3. Explore visualizations in the different tabs
            4. Save results for further analysis
            
            ### Symmetry Types:
            
            - **Mirror Symmetry**: Reflection across an axis
            - **Time-Reversal Symmetry**: Running the trajectory backwards
            - **Rotational Symmetry**: Rotating the state space
            
            ### References:
            
            - [Hu & Dear, 2023] Guided Deep RL for Articulated Swimming Robots
            - [Pujari et al., 2023] Detecting and Exploiting Symmetry to Accelerate RL
            """)


if __name__ == "__main__":
    main() 