"""
VLA-Lab Streamlit Multi-Page Application

Main entry point for the visualization app.
Usage: streamlit run app.py
"""

import streamlit as st
from pathlib import Path

# Page config must be the first Streamlit command
st.set_page_config(
    page_title="VLA-Lab",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Setup matplotlib fonts
try:
    from vlalab.viz.mpl_fonts import setup_matplotlib_fonts
    setup_matplotlib_fonts(verbose=False)
except Exception:
    pass


def main():
    st.sidebar.title("🤖 VLA-Lab")
    st.sidebar.markdown("---")
    
    # Navigation
    pages = {
        "🏠 Home": "home",
        "🔬 Inference Runs": "inference",
        "📊 Dataset Viewer": "dataset",
        "📈 Latency Analysis": "latency",
        "🎯 Open-Loop Eval": "eval",
    }
    
    selection = st.sidebar.radio("Navigate", list(pages.keys()))
    
    page_name = pages[selection]
    
    if page_name == "home":
        show_home_page()
    elif page_name == "inference":
        from vlalab.apps.streamlit.pages import inference_viewer
        inference_viewer.render()
    elif page_name == "dataset":
        from vlalab.apps.streamlit.pages import dataset_viewer
        dataset_viewer.render()
    elif page_name == "latency":
        from vlalab.apps.streamlit.pages import latency_viewer
        latency_viewer.render()
    elif page_name == "eval":
        from vlalab.apps.streamlit.pages import eval_viewer
        eval_viewer.render()
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.caption("VLA-Lab v0.1.0")


def show_home_page():
    st.title("🤖 VLA-Lab")
    st.markdown("""
    **A toolbox for tracking and visualizing the real-world deployment process of VLA models.**
    
    ### Features
    
    - **🔬 Inference Run Viewer**: Replay and analyze policy inference sessions
      - Step-by-step visualization with multi-camera support
      - 3D trajectory and action visualization
      - Latency breakdown analysis
    
    - **📊 Dataset Viewer**: Browse and analyze training/evaluation datasets
      - Zarr dataset support (Diffusion Policy format)
      - Episode navigation with image grid view
      - Action trajectory analysis
    
    - **📈 Latency Analysis**: Deep dive into timing metrics
      - Transport latency (network)
      - Inference latency (GPU)
      - End-to-end loop time
    
    - **🎯 Open-Loop Eval**: Evaluate model predictions vs ground truth
      - Compare predicted actions with dataset actions
      - MSE/MAE metrics per trajectory
      - Multi-dimensional action visualization
    
    ### Supported Frameworks
    
    | Framework | Status |
    |-----------|--------|
    | RealWorld-DP (Diffusion Policy) | ✅ Supported |
    | Isaac-GR00T | ✅ Supported |
    
    ### Quick Start
    
    1. **View inference logs**: Select "🔬 Inference Runs" from the sidebar
    2. **Browse datasets**: Select "📊 Dataset Viewer" from the sidebar
    3. **Analyze latency**: Select "📈 Latency Analysis" from the sidebar
    
    ---
    
    📖 [Documentation](https://github.com/VLA-Lab/VLA-Lab) | 
    🐛 [Report Issues](https://github.com/VLA-Lab/VLA-Lab/issues)
    """)


if __name__ == "__main__":
    main()
