"""
VLA-Lab Open-Loop Evaluation Viewer

Visualize and compare predicted vs ground-truth actions from open-loop evaluation.
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from typing import Dict, List, Any, Optional

# Setup matplotlib fonts
try:
    from vlalab.viz.mpl_fonts import setup_matplotlib_fonts
    setup_matplotlib_fonts(verbose=False)
except Exception:
    pass


def load_eval_results(results_path: Path) -> Dict[str, Any]:
    """Load evaluation results from JSON file."""
    with open(results_path, "r") as f:
        return json.load(f)


def load_trajectory_arrays(results_dir: Path, traj_id: int) -> Dict[str, np.ndarray]:
    """
    Load trajectory arrays (GT and pred actions) if saved as .npy files.
    
    This is a placeholder - actual implementation depends on how arrays are saved.
    """
    arrays = {}
    
    gt_path = results_dir / f"traj_{traj_id}_gt.npy"
    pred_path = results_dir / f"traj_{traj_id}_pred.npy"
    
    if gt_path.exists():
        arrays["gt_actions"] = np.load(gt_path)
    if pred_path.exists():
        arrays["pred_actions"] = np.load(pred_path)
    
    return arrays


def plot_action_comparison(
    gt_actions: np.ndarray,
    pred_actions: np.ndarray,
    dim_idx: int,
    action_horizon: int = 16,
    dim_label: str = "",
) -> plt.Figure:
    """Plot GT vs predicted actions for a single dimension."""
    fig, ax = plt.subplots(figsize=(12, 4))
    
    steps = np.arange(len(gt_actions))
    
    ax.plot(steps, gt_actions[:, dim_idx], label="Ground Truth", alpha=0.8, linewidth=2)
    ax.plot(steps, pred_actions[:, dim_idx], label="Predicted", alpha=0.8, linewidth=2, linestyle="--")
    
    # Mark inference points
    for j in range(0, len(gt_actions), action_horizon):
        ax.axvline(x=j, color='gray', linestyle=':', alpha=0.3)
    
    # Error band
    error = np.abs(gt_actions[:, dim_idx] - pred_actions[:, dim_idx])
    ax.fill_between(steps, 
                    gt_actions[:, dim_idx] - error,
                    gt_actions[:, dim_idx] + error,
                    alpha=0.2, color='red', label='Error')
    
    ax.set_xlabel("Step")
    ax.set_ylabel("Value")
    ax.set_title(f"Action Dimension: {dim_label or dim_idx}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig


def plot_all_dimensions(
    gt_actions: np.ndarray,
    pred_actions: np.ndarray,
    action_horizon: int = 16,
    action_labels: Optional[List[str]] = None,
) -> plt.Figure:
    """Plot all action dimensions in a grid."""
    num_dims = gt_actions.shape[1]
    n_cols = min(3, num_dims)
    n_rows = (num_dims + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3 * n_rows))
    axes = np.atleast_2d(axes)
    
    steps = np.arange(len(gt_actions))
    
    for i in range(num_dims):
        row, col = i // n_cols, i % n_cols
        ax = axes[row, col]
        
        ax.plot(steps, gt_actions[:, i], label="GT", alpha=0.8)
        ax.plot(steps, pred_actions[:, i], label="Pred", alpha=0.8, linestyle="--")
        
        label = action_labels[i] if action_labels and i < len(action_labels) else f"Dim {i}"
        ax.set_title(label, fontsize=10)
        ax.grid(True, alpha=0.3)
        
        if i == 0:
            ax.legend(fontsize=8)
    
    # Hide empty subplots
    for i in range(num_dims, n_rows * n_cols):
        row, col = i // n_cols, i % n_cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    return fig


def plot_3d_trajectory(
    gt_actions: np.ndarray,
    pred_actions: np.ndarray,
    xyz_dims: List[int] = [0, 1, 2],
) -> plt.Figure:
    """Plot 3D trajectory comparison (if actions contain position)."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    x, y, z = xyz_dims
    
    # GT trajectory
    ax.plot(gt_actions[:, x], gt_actions[:, y], gt_actions[:, z],
            'b-', label='Ground Truth', alpha=0.8, linewidth=2)
    ax.scatter(gt_actions[0, x], gt_actions[0, y], gt_actions[0, z],
               c='green', s=100, label='Start')
    ax.scatter(gt_actions[-1, x], gt_actions[-1, y], gt_actions[-1, z],
               c='red', s=100, label='End')
    
    # Pred trajectory
    ax.plot(pred_actions[:, x], pred_actions[:, y], pred_actions[:, z],
            'r--', label='Predicted', alpha=0.6, linewidth=2)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    ax.set_title('3D Trajectory Comparison')
    
    return fig


def plot_error_histogram(
    gt_actions: np.ndarray,
    pred_actions: np.ndarray,
) -> plt.Figure:
    """Plot error distribution histogram."""
    errors = (gt_actions - pred_actions).flatten()
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Error histogram
    axes[0].hist(errors, bins=50, alpha=0.7, edgecolor='black')
    axes[0].axvline(x=0, color='r', linestyle='--', label='Zero')
    axes[0].set_xlabel('Error')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Error Distribution')
    axes[0].legend()
    
    # Absolute error histogram
    abs_errors = np.abs(errors)
    axes[1].hist(abs_errors, bins=50, alpha=0.7, edgecolor='black', color='orange')
    axes[1].axvline(x=np.mean(abs_errors), color='r', linestyle='--', 
                    label=f'Mean: {np.mean(abs_errors):.4f}')
    axes[1].set_xlabel('Absolute Error')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Absolute Error Distribution')
    axes[1].legend()
    
    plt.tight_layout()
    return fig


def render():
    """Render the evaluation viewer page."""
    st.title("📊 Open-Loop 评估结果")
    
    st.markdown("""
    可视化 VLA 模型的 Open-Loop 评估结果，比较预测动作与真实动作。
    
    **使用方法:**
    1. 上传评估结果 JSON 文件
    2. 或指定包含评估图片的目录
    """)
    
    # Sidebar options
    st.sidebar.markdown("### 数据来源")
    
    source = st.sidebar.radio(
        "选择数据来源",
        ["上传 JSON", "浏览目录", "演示数据"],
    )
    
    results = None
    gt_actions = None
    pred_actions = None
    
    if source == "上传 JSON":
        uploaded_file = st.sidebar.file_uploader(
            "上传评估结果 JSON",
            type=["json"],
        )
        
        if uploaded_file:
            results = json.load(uploaded_file)
            st.success(f"已加载评估结果: {len(results.get('results', []))} 条轨迹")
    
    elif source == "浏览目录":
        results_dir = st.sidebar.text_input(
            "评估结果目录",
            value="",
            placeholder="/path/to/eval_results/",
        )
        
        if results_dir and Path(results_dir).exists():
            # Find JSON files
            json_files = list(Path(results_dir).glob("*.json"))
            if json_files:
                selected_json = st.sidebar.selectbox(
                    "选择结果文件",
                    json_files,
                    format_func=lambda p: p.name,
                )
                if selected_json:
                    results = load_eval_results(selected_json)
            
            # Find plot images
            png_files = sorted(Path(results_dir).glob("*.png"))
            if png_files:
                st.markdown("### 已生成的评估图")
                for png_file in png_files:
                    st.image(str(png_file), caption=png_file.name)
    
    elif source == "演示数据":
        st.info("使用随机生成的演示数据")
        
        # Generate demo data
        np.random.seed(42)
        num_steps = 200
        action_dim = 8
        
        # Simulate GT actions (smooth trajectory)
        t = np.linspace(0, 4 * np.pi, num_steps)
        gt_actions = np.zeros((num_steps, action_dim))
        gt_actions[:, 0] = 0.5 * np.sin(t) + 0.1 * np.random.randn(num_steps)
        gt_actions[:, 1] = 0.3 * np.cos(t) + 0.1 * np.random.randn(num_steps)
        gt_actions[:, 2] = 0.2 * np.sin(0.5 * t) + 0.1 * np.random.randn(num_steps)
        for i in range(3, action_dim):
            gt_actions[:, i] = 0.1 * np.sin(t * (i - 2)) + 0.05 * np.random.randn(num_steps)
        
        # Simulate pred actions (GT + noise + slight bias)
        pred_actions = gt_actions + 0.05 * np.random.randn(num_steps, action_dim)
        pred_actions += 0.02 * np.ones_like(pred_actions)  # slight bias
        
        # Calculate metrics
        mse = float(np.mean((gt_actions - pred_actions) ** 2))
        mae = float(np.mean(np.abs(gt_actions - pred_actions)))
        
        results = {
            "results": [{"trajectory_id": 0, "mse": mse, "mae": mae, "num_steps": num_steps}],
            "avg_mse": mse,
            "avg_mae": mae,
            "num_trajectories": 1,
        }
    
    # Display results
    if results:
        st.markdown("---")
        
        # Summary metrics
        st.markdown("### 📈 评估指标")
        col1, col2, col3 = st.columns(3)
        
        col1.metric("平均 MSE", f"{results.get('avg_mse', 0):.6f}")
        col2.metric("平均 MAE", f"{results.get('avg_mae', 0):.6f}")
        col3.metric("评估轨迹数", results.get('num_trajectories', 0))
        
        # Per-trajectory results
        if results.get("results"):
            st.markdown("### 📋 轨迹详情")
            
            traj_data = []
            for r in results["results"]:
                traj_data.append({
                    "轨迹 ID": r["trajectory_id"],
                    "MSE": f"{r['mse']:.6f}",
                    "MAE": f"{r['mae']:.6f}",
                    "步数": r["num_steps"],
                })
            
            st.dataframe(traj_data, use_container_width=True)
    
    # Visualization tabs
    if gt_actions is not None and pred_actions is not None:
        st.markdown("---")
        st.markdown("### 🎨 可视化")
        
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 时序对比",
            "🗺️ 3D 轨迹",
            "📈 误差分布",
            "🔍 逐维度分析",
        ])
        
        with tab1:
            action_horizon = st.slider("Action Horizon", 4, 32, 16)
            fig = plot_all_dimensions(gt_actions, pred_actions, action_horizon)
            st.pyplot(fig)
            plt.close(fig)
        
        with tab2:
            if gt_actions.shape[1] >= 3:
                st.markdown("假设前三个维度为 XYZ 位置")
                fig = plot_3d_trajectory(gt_actions, pred_actions)
                st.pyplot(fig)
                plt.close(fig)
            else:
                st.warning("动作维度不足，无法绘制 3D 轨迹")
        
        with tab3:
            fig = plot_error_histogram(gt_actions, pred_actions)
            st.pyplot(fig)
            plt.close(fig)
        
        with tab4:
            dim_idx = st.selectbox(
                "选择维度",
                range(gt_actions.shape[1]),
                format_func=lambda x: f"维度 {x}",
            )
            fig = plot_action_comparison(gt_actions, pred_actions, dim_idx)
            st.pyplot(fig)
            plt.close(fig)
    
    # Usage instructions
    with st.expander("💡 如何生成评估结果"):
        st.markdown("""
        ```python
        from vlalab.eval import OpenLoopEvaluator
        from vlalab.eval.adapters import GR00TAdapter
        
        # 1. 创建 Policy 适配器
        adapter = GR00TAdapter(policy)
        
        # 2. 创建评估器
        evaluator = OpenLoopEvaluator(
            policy=adapter,
            dataset_path="/path/to/dataset.zarr",
        )
        
        # 3. 运行评估
        results = evaluator.evaluate(
            traj_ids=[0, 1, 2],
            max_steps=200,
            save_plots_dir="eval_outputs/",
        )
        
        # 4. 保存结果
        evaluator.evaluate_and_save(
            "eval_outputs/results.json",
            traj_ids=[0, 1, 2],
        )
        ```
        """)
