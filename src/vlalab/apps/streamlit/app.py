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

# Custom CSS for styling and hiding default navigation
st.markdown("""
<style>
/* Hide default Streamlit page navigation */
[data-testid="stSidebarNav"] {
    display: none !important;
}

/* Hide hamburger menu on pages */
header[data-testid="stHeader"] {
    background: transparent;
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
}

[data-testid="stSidebar"] .stMarkdown,
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3,
[data-testid="stSidebar"] .stMarkdown h4 {
    color: #e8e8e8 !important;
}

[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stTextInput label,
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stMultiSelect label {
    color: #d0d0d0 !important;
}

[data-testid="stSidebar"] .stCaption,
[data-testid="stSidebar"] small {
    color: #a0a0a0 !important;
}

/* Sidebar title */
.sidebar-title {
    font-size: 1.8rem;
    font-weight: 700;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.5rem;
    letter-spacing: -0.5px;
}

.sidebar-subtitle {
    color: #a0a0a0;
    font-size: 0.85rem;
    margin-bottom: 1.5rem;
}

/* Navigation styling */
[data-testid="stSidebar"] .stRadio > label {
    color: #ffffff !important;
    font-weight: 500;
}

[data-testid="stSidebar"] .stRadio > div {
    gap: 0.3rem;
}

[data-testid="stSidebar"] .stRadio > div > label {
    padding: 0.6rem 0.8rem;
    border-radius: 8px;
    transition: all 0.2s ease;
    cursor: pointer;
    color: #e8e8e8 !important;
}

[data-testid="stSidebar"] .stRadio > div > label span,
[data-testid="stSidebar"] .stRadio > div > label p {
    color: #e8e8e8 !important;
}

[data-testid="stSidebar"] .stRadio > div > label:hover {
    background: rgba(102, 126, 234, 0.25);
}

[data-testid="stSidebar"] .stRadio > div > label[data-checked="true"] {
    background: linear-gradient(135deg, rgba(102, 126, 234, 0.4) 0%, rgba(118, 75, 162, 0.4) 100%);
    border-left: 3px solid #667eea;
}

[data-testid="stSidebar"] .stRadio > div > label[data-checked="true"] span,
[data-testid="stSidebar"] .stRadio > div > label[data-checked="true"] p {
    color: #ffffff !important;
    font-weight: 600;
}

/* Main content area */
.main .block-container {
    padding-top: 2rem;
    max-width: 1400px;
}

/* Home page styling */
.hero-title {
    font-size: 3rem;
    font-weight: 800;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.5rem;
}

.hero-subtitle {
    font-size: 1.2rem;
    color: #666;
    margin-bottom: 2rem;
}

/* Feature cards */
.feature-card {
    background: linear-gradient(135deg, #f5f7fa 0%, #e4e8eb 100%);
    border-radius: 12px;
    padding: 1.5rem;
    margin: 0.5rem 0;
    border-left: 4px solid #667eea;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}

.feature-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(102, 126, 234, 0.15);
}

.feature-card h4 {
    color: #1a1a2e;
    margin-bottom: 0.5rem;
}

.feature-card p {
    color: #555;
    font-size: 0.9rem;
    margin: 0;
}

/* Status badges */
.status-badge {
    display: inline-block;
    padding: 0.2rem 0.6rem;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
}

.status-supported {
    background: #d4edda;
    color: #155724;
}

/* Footer */
.sidebar-footer {
    color: #666;
    font-size: 0.75rem;
    padding: 1rem 0;
    border-top: 1px solid rgba(255,255,255,0.1);
    margin-top: 2rem;
}

/* Tabs styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
}

.stTabs [data-baseweb="tab"] {
    border-radius: 8px 8px 0 0;
    padding: 0.5rem 1rem;
}

/* Metrics */
[data-testid="stMetric"] {
    background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
    padding: 1rem;
    border-radius: 10px;
    border: 1px solid #dee2e6;
}

/* Info boxes */
.stAlert {
    border-radius: 10px;
}

/* Dataframe */
.stDataFrame {
    border-radius: 10px;
    overflow: hidden;
}
</style>
""", unsafe_allow_html=True)

# Setup matplotlib fonts
try:
    from vlalab.viz.mpl_fonts import setup_matplotlib_fonts
    setup_matplotlib_fonts(verbose=False)
except Exception:
    pass


def main():
    # Sidebar header
    st.sidebar.markdown("""
        <div class="sidebar-title">🤖 VLA-Lab</div>
        <div class="sidebar-subtitle">VLA 部署追踪与可视化</div>
    """, unsafe_allow_html=True)
    
    # Navigation
    pages = {
        "🏠 首页": "home",
        "🔬 推理回放": "inference",
        "📊 数据集浏览": "dataset",
        "📈 延迟分析": "latency",
        "🎯 开环评估": "eval",
    }
    
    selection = st.sidebar.radio(
        "导航",
        list(pages.keys()),
        label_visibility="collapsed",
    )
    
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
    st.sidebar.markdown("""
        <div class="sidebar-footer">
            VLA-Lab v0.1.0<br>
            <a href="https://github.com/VLA-Lab/VLA-Lab" style="color: #667eea;">GitHub</a>
        </div>
    """, unsafe_allow_html=True)


def show_home_page():
    # Hero section
    st.markdown("""
        <div class="hero-title">VLA-Lab</div>
        <div class="hero-subtitle">
            专为 VLA (Vision-Language-Action) 模型设计的实机部署追踪与可视化工具箱
        </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    
    # Features section
    st.markdown("### ✨ 核心功能")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h4>🔬 推理回放 Inference Replay</h4>
            <p>
                逐帧回放策略推理过程，支持多相机视角、3D 轨迹可视化、延迟诊断。
                快速定位部署问题。
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card">
            <h4>📊 数据集浏览 Dataset Viewer</h4>
            <p>
                可视化 Zarr 格式训练数据集，支持 Episode 导航、图像网格、动作轨迹分析。
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <h4>📈 延迟分析 Latency Analysis</h4>
            <p>
                深度分析控制回路时延：传输延迟、推理延迟、端到端回路时间。
                多运行对比，识别性能瓶颈。
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card">
            <h4>🎯 开环评估 Open-Loop Eval</h4>
            <p>
                对比模型预测动作与真实动作，计算 MSE/MAE 指标，
                多维度动作可视化。
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("### 🔧 支持的框架")
    
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        st.markdown("""
        **RealWorld-DP**  
        <span class="status-badge status-supported">✅ 已支持</span>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        **Isaac-GR00T**  
        <span class="status-badge status-supported">✅ 已支持</span>
        """, unsafe_allow_html=True)
    
    st.markdown("### 🚀 快速开始")
    
    st.code("""
# 1. 安装
pip install vlalab

# 2. 在推理代码中集成日志
import vlalab

logger = vlalab.init(
    project="my_project",
    model="gr00t-n1",
    task="pick_and_place",
)

# 3. 记录每步数据
logger.log_step(
    obs={"images": [img], "state": state},
    action={"values": action},
    timing={"inference_latency_ms": latency},
)

# 4. 启动可视化
# vlalab view
""", language="python")
    
    st.info("👈 从左侧导航栏选择功能开始使用")


if __name__ == "__main__":
    main()
