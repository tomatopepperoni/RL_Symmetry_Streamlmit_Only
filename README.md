# 🤖 RL Symmetry Visualization Dashboard

An interactive web dashboard for analyzing symmetry patterns in reinforcement learning trajectories.

## 🌐 Live Demo

Visit the live dashboard: [Coming Soon]

## ✨ Features

- **Interactive Trajectory Visualization**: Explore agent movements with customizable dimension selection
- **Symmetry Analysis**: Detect and visualize geometric transformations (mirror, rotation, time-reversal)
- **Behavior Clustering**: Group similar behaviors and detect anomalies using PCA and k-means
- **Multi-Dataset Comparison**: Compare multiple experiments side-by-side
- **Real-time Controls**: Adjust parameters and see updates instantly

## 🚀 Quick Start

### Run Locally

```bash
# Clone the repository
git clone https://github.com/tomatopepperoni/RL_Symmetry_Streamlmit_Only.git
cd RL_Symmetry_Streamlmit_Only

# Install dependencies
pip install -r requirements.txt

# Run the dashboard
streamlit run app.py
```

The dashboard will open in your browser at `http://localhost:8501`

## 📊 How to Use

1. **Load Data**: Upload CSV trajectory files or generate demo data
2. **Explore Trajectories**: Select dimensions and time ranges to visualize
3. **Analyze Symmetry**: Choose transformations and run symmetry detection
4. **Cluster Behaviors**: Use k-means or DBSCAN to group similar behaviors
5. **Compare Experiments**: Load multiple datasets for side-by-side comparison

## 📁 Data Format

Upload CSV files with the following column naming convention:
- `state_0`, `state_1`, ... for state dimensions
- `action_0`, `action_1`, ... for action dimensions
- `reward` for reward values

## 🎯 Research Context

This dashboard supports research in symmetry-aware reinforcement learning for robotic locomotion, based on:
- Detecting and Exploiting Symmetry to Accelerate Reinforcement Learning
- Exploiting Symmetry to Accelerate Reinforcement Learning

## 🛠️ Technology Stack

- **Frontend**: Streamlit
- **Visualization**: Plotly, Matplotlib
- **Analysis**: NumPy, Pandas, scikit-learn
- **Deployment**: Streamlit Community Cloud

## 📝 License

MIT License - see LICENSE file for details

## 🤝 Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.

---

Built with ❤️ for the RL research community

