"""
VLA-Lab Open-Loop Evaluation Viewer

Interactive visualization for open-loop evaluation results.
Uses Plotly for rich interactive charts instead of static images.
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import json
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_eval_results(results_path: Path) -> Dict[str, Any]:
    """Load evaluation results from JSON file."""
    with open(results_path, "r") as f:
        return json.load(f)


def discover_trajectories(results_dir: Path) -> List[int]:
    """Discover trajectory IDs from saved npy files in a directory."""
    traj_ids = set()
    for f in results_dir.glob("traj_*_gt.npy"):
        parts = f.stem.replace("traj_", "").replace("_gt", "")
        try:
            traj_ids.add(int(parts))
        except ValueError:
            pass
    return sorted(traj_ids)


def load_trajectory_arrays(
    results_dir: Path, traj_id: int
) -> Dict[str, np.ndarray]:
    """Load GT/pred/state arrays for a trajectory."""
    arrays: Dict[str, np.ndarray] = {}
    for suffix, key in [("gt", "gt_actions"), ("pred", "pred_actions"), ("states", "states")]:
        p = results_dir / f"traj_{traj_id}_{suffix}.npy"
        if p.exists():
            arrays[key] = np.load(p)
    return arrays


def generate_demo_data() -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Generate demo data for preview."""
    np.random.seed(42)
    T, D = 200, 8
    t = np.linspace(0, 4 * np.pi, T)
    gt = np.column_stack([
        0.5 * np.sin(t),
        0.3 * np.cos(t),
        0.2 * np.sin(0.5 * t),
        *(0.1 * np.sin(t * (i - 2)) for i in range(3, D)),
    ]) + 0.02 * np.random.randn(T, D)
    pred = gt + 0.05 * np.random.randn(T, D) + 0.02
    mse = float(np.mean((gt - pred) ** 2))
    mae = float(np.mean(np.abs(gt - pred)))
    meta = {
        "results": [{"trajectory_id": 0, "mse": mse, "mae": mae, "num_steps": T}],
        "avg_mse": mse,
        "avg_mae": mae,
        "num_trajectories": 1,
        "action_keys": ["x", "y", "z", "qx", "qy", "qz", "qw", "gripper"],
        "action_horizon": 16,
    }
    return gt, pred, meta


# ---------------------------------------------------------------------------
# Plotly chart builders
# ---------------------------------------------------------------------------

def build_timeseries_chart(
    gt: np.ndarray,
    pred: np.ndarray,
    action_keys: List[str],
    action_horizon: int,
    selected_dims: Optional[List[int]] = None,
) -> go.Figure:
    """Interactive time-series comparison of GT vs Pred actions."""
    dims = selected_dims if selected_dims is not None else list(range(gt.shape[1]))
    n_dims = len(dims)
    labels = [action_keys[d] if d < len(action_keys) else f"dim_{d}" for d in dims]

    fig = make_subplots(
        rows=n_dims, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=[f"Action: {lb}" for lb in labels],
    )

    T = len(gt)
    for row_idx, d in enumerate(dims, 1):
        lb = labels[row_idx - 1]
        # GT
        fig.add_trace(
            go.Scatter(
                y=gt[:, d], mode="lines", name=f"GT {lb}",
                line=dict(width=1.8),
                legendgroup="gt", legendgrouptitle_text="Ground Truth",
                showlegend=(row_idx == 1),
                hovertemplate=f"GT {lb}: %{{y:.4f}}<extra></extra>",
            ),
            row=row_idx, col=1,
        )
        # Pred
        fig.add_trace(
            go.Scatter(
                y=pred[:, d], mode="lines", name=f"Pred {lb}",
                line=dict(width=1.8, dash="dot"),
                legendgroup="pred", legendgrouptitle_text="Predicted",
                showlegend=(row_idx == 1),
                hovertemplate=f"Pred {lb}: %{{y:.4f}}<extra></extra>",
            ),
            row=row_idx, col=1,
        )
        # Inference point markers
        inf_x = list(range(0, T, action_horizon))
        inf_y = [float(gt[j, d]) for j in inf_x]
        fig.add_trace(
            go.Scatter(
                x=inf_x, y=inf_y, mode="markers",
                marker=dict(color="red", size=5, symbol="circle"),
                name="Inference Point",
                legendgroup="inf",
                showlegend=(row_idx == 1),
                hoverinfo="skip",
            ),
            row=row_idx, col=1,
        )

    fig.update_layout(
        height=max(280 * n_dims, 400),
        hovermode="x unified",
        template="plotly_white",
        margin=dict(l=40, r=20, t=30, b=40),
        legend=dict(orientation="h", y=-0.05, x=0, xanchor="left"),
    )
    fig.update_xaxes(title_text="Step", row=n_dims, col=1)
    return fig


def build_error_timeseries(
    gt: np.ndarray,
    pred: np.ndarray,
    action_keys: List[str],
    action_horizon: int,
) -> go.Figure:
    """Per-dimension absolute error over time."""
    D = gt.shape[1]
    abs_err = np.abs(gt - pred)  # (T, D)

    fig = go.Figure()
    for d in range(D):
        lb = action_keys[d] if d < len(action_keys) else f"dim_{d}"
        fig.add_trace(go.Scatter(
            y=abs_err[:, d], mode="lines", name=lb,
            line=dict(width=1.5),
            hovertemplate=f"{lb}: %{{y:.4f}}<extra></extra>",
        ))

    # Total error
    fig.add_trace(go.Scatter(
        y=np.mean(abs_err, axis=1), mode="lines", name="Mean(all dims)",
        line=dict(width=2.5, color="black", dash="dash"),
        hovertemplate="Mean: %{y:.4f}<extra></extra>",
    ))

    fig.update_layout(
        height=400,
        title="逐维度绝对误差随时间变化",
        xaxis_title="Step", yaxis_title="Absolute Error",
        hovermode="x unified",
        template="plotly_white",
        legend=dict(orientation="h", y=-0.2, x=0),
    )
    return fig


def build_per_dim_metrics_bar(
    gt: np.ndarray,
    pred: np.ndarray,
    action_keys: List[str],
) -> go.Figure:
    """Bar chart of per-dimension MSE and MAE."""
    D = gt.shape[1]
    labels = [action_keys[d] if d < len(action_keys) else f"dim_{d}" for d in range(D)]
    mse_per_dim = [float(np.mean((gt[:, d] - pred[:, d]) ** 2)) for d in range(D)]
    mae_per_dim = [float(np.mean(np.abs(gt[:, d] - pred[:, d]))) for d in range(D)]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=labels, y=mse_per_dim, name="MSE",
        marker_color="rgba(99,110,250,0.8)",
        hovertemplate="%{x}: %{y:.6f}<extra>MSE</extra>",
    ))
    fig.add_trace(go.Bar(
        x=labels, y=mae_per_dim, name="MAE",
        marker_color="rgba(239,85,59,0.8)",
        hovertemplate="%{x}: %{y:.6f}<extra>MAE</extra>",
    ))

    fig.update_layout(
        barmode="group",
        title="逐维度 MSE / MAE",
        xaxis_title="Action Dimension",
        yaxis_title="Error",
        height=350,
        template="plotly_white",
        legend=dict(orientation="h", y=1.12, x=0.5, xanchor="center"),
    )
    return fig


def build_error_heatmap(
    gt: np.ndarray,
    pred: np.ndarray,
    action_keys: List[str],
    action_horizon: int,
) -> go.Figure:
    """Heatmap of absolute error (X=step, Y=dimension)."""
    abs_err = np.abs(gt - pred).T  # (D, T)
    D = abs_err.shape[0]
    labels = [action_keys[d] if d < len(action_keys) else f"dim_{d}" for d in range(D)]

    fig = go.Figure(go.Heatmap(
        z=abs_err,
        y=labels,
        colorscale="YlOrRd",
        colorbar_title="Abs Error",
        hovertemplate="Step %{x}<br>%{y}<br>Error: %{z:.4f}<extra></extra>",
    ))

    # Add inference-point vertical lines
    T = abs_err.shape[1]
    for j in range(0, T, action_horizon):
        fig.add_vline(x=j, line_width=0.5, line_dash="dot", line_color="gray", opacity=0.4)

    fig.update_layout(
        title="误差热力图 (越红误差越大)",
        xaxis_title="Step",
        yaxis_title="Action Dimension",
        height=max(250, 50 * D),
        template="plotly_white",
    )
    return fig


def build_3d_trajectory(
    gt: np.ndarray,
    pred: np.ndarray,
    xyz_dims: Tuple[int, int, int] = (0, 1, 2),
    action_keys: Optional[List[str]] = None,
) -> go.Figure:
    """3D trajectory comparison."""
    x, y, z = xyz_dims
    lx = action_keys[x] if action_keys and x < len(action_keys) else "X"
    ly = action_keys[y] if action_keys and y < len(action_keys) else "Y"
    lz = action_keys[z] if action_keys and z < len(action_keys) else "Z"

    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=gt[:, x], y=gt[:, y], z=gt[:, z],
        mode="lines", name="GT",
        line=dict(color="royalblue", width=4),
    ))
    fig.add_trace(go.Scatter3d(
        x=pred[:, x], y=pred[:, y], z=pred[:, z],
        mode="lines", name="Pred",
        line=dict(color="tomato", width=3, dash="dot"),
    ))
    # Start / end
    fig.add_trace(go.Scatter3d(
        x=[gt[0, x]], y=[gt[0, y]], z=[gt[0, z]],
        mode="markers", name="Start",
        marker=dict(size=6, color="green", symbol="diamond"),
    ))
    fig.add_trace(go.Scatter3d(
        x=[gt[-1, x]], y=[gt[-1, y]], z=[gt[-1, z]],
        mode="markers", name="End",
        marker=dict(size=6, color="red", symbol="diamond"),
    ))

    fig.update_layout(
        height=550,
        scene=dict(xaxis_title=lx, yaxis_title=ly, zaxis_title=lz, aspectmode="data"),
        template="plotly_white",
        title="3D 轨迹对比 (GT vs Predicted)",
        legend=dict(orientation="h", y=-0.05),
    )
    return fig


def build_error_distribution(
    gt: np.ndarray,
    pred: np.ndarray,
    action_keys: List[str],
) -> go.Figure:
    """Error distribution violin + box plot per dimension."""
    D = gt.shape[1]
    fig = go.Figure()
    for d in range(D):
        lb = action_keys[d] if d < len(action_keys) else f"dim_{d}"
        err = gt[:, d] - pred[:, d]
        fig.add_trace(go.Violin(
            y=err, name=lb,
            box_visible=True,
            meanline_visible=True,
            hoverinfo="y",
        ))

    fig.update_layout(
        title="逐维度误差分布 (Violin + Box)",
        yaxis_title="Error (GT - Pred)",
        height=400,
        template="plotly_white",
        showlegend=False,
    )
    return fig


# ---------------------------------------------------------------------------
# Main render
# ---------------------------------------------------------------------------

def render():
    """Render the evaluation viewer page."""
    st.title("🎯 Open-Loop 评估可视化")

    # ── Sidebar: data source ──
    st.sidebar.markdown("### 📂 数据来源")
    source = st.sidebar.radio(
        "选择来源",
        ["浏览目录", "上传 JSON", "演示数据"],
        label_visibility="collapsed",
    )

    results: Optional[Dict[str, Any]] = None
    gt_actions: Optional[np.ndarray] = None
    pred_actions: Optional[np.ndarray] = None
    results_dir: Optional[Path] = None
    available_traj_ids: List[int] = []

    # ── Load data ──
    if source == "浏览目录":
        dir_path = st.sidebar.text_input(
            "评估结果目录",
            value="",
            placeholder="/path/to/eval_outputs/",
        )
        if dir_path and Path(dir_path).exists():
            results_dir = Path(dir_path)
            # Load JSON
            json_files = sorted(results_dir.glob("*.json"))
            if json_files:
                selected_json = st.sidebar.selectbox(
                    "结果文件", json_files, format_func=lambda p: p.name,
                )
                if selected_json:
                    results = load_eval_results(selected_json)
            # Discover trajectories
            available_traj_ids = discover_trajectories(results_dir)
        else:
            if dir_path:
                st.sidebar.error("路径不存在")

    elif source == "上传 JSON":
        uploaded = st.sidebar.file_uploader("上传 results.json", type=["json"])
        if uploaded:
            results = json.load(uploaded)
            st.sidebar.success(f"已加载 {len(results.get('results', []))} 条轨迹")

    elif source == "演示数据":
        gt_actions, pred_actions, results = generate_demo_data()
        available_traj_ids = [0]

    # ── Nothing loaded yet ──
    if results is None and gt_actions is None:
        st.info("👈 请在左侧选择数据来源并指定评估结果目录。")
        _show_usage_guide()
        return

    # ── Extract meta from results ──
    action_keys: List[str] = results.get("action_keys", []) if results else []
    action_horizon: int = results.get("action_horizon", 16) if results else 16

    # ── Summary metrics ──
    if results:
        st.markdown("### 📊 评估总览")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("轨迹数", results.get("num_trajectories", 0))
        c2.metric("平均 MSE", f"{results.get('avg_mse', 0):.6f}")
        c3.metric("平均 MAE", f"{results.get('avg_mae', 0):.6f}")
        c4.metric("Action Horizon", action_horizon)

        # Per-trajectory table
        if results.get("results"):
            import pandas as pd
            df = pd.DataFrame(results["results"])
            df.columns = [c.replace("_", " ").title() for c in df.columns]
            st.dataframe(df, use_container_width=True, hide_index=True)

    # ── Trajectory selector ──
    if results_dir and available_traj_ids:
        selected_traj = st.sidebar.selectbox(
            "选择轨迹", available_traj_ids,
            format_func=lambda x: f"Trajectory {x}",
        )
        arrays = load_trajectory_arrays(results_dir, selected_traj)
        gt_actions = arrays.get("gt_actions")
        pred_actions = arrays.get("pred_actions")
        if gt_actions is None or pred_actions is None:
            st.warning(f"轨迹 {selected_traj} 的 npy 数组文件不存在，只能查看汇总指标。")

    # ── No arrays → only show summary ──
    if gt_actions is None or pred_actions is None:
        if results_dir:
            # fallback: show saved PNG images
            png_files = sorted(results_dir.glob("traj_*.png"))
            if png_files:
                st.markdown("---")
                st.markdown("### 🖼️ 评估图 (静态)")
                for pf in png_files:
                    st.image(str(pf), caption=pf.stem)
        return

    # ── Infer action_keys if missing ──
    D = gt_actions.shape[1]
    if not action_keys or len(action_keys) != D:
        action_keys = [f"dim_{i}" for i in range(D)]

    # ── Sidebar: dimension filter ──
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔧 显示设置")
    all_dims = list(range(D))
    selected_dims = st.sidebar.multiselect(
        "选择维度",
        all_dims,
        default=all_dims,
        format_func=lambda d: action_keys[d] if d < len(action_keys) else f"dim_{d}",
    )
    if not selected_dims:
        selected_dims = all_dims

    # ── Tabs ──
    st.markdown("---")
    tabs = st.tabs([
        "📈 时序对比",
        "🔥 误差热力图",
        "📊 维度分析",
        "🗺️ 3D 轨迹",
        "🎻 误差分布",
    ])

    # Tab 1: Time-series comparison
    with tabs[0]:
        fig = build_timeseries_chart(gt_actions, pred_actions, action_keys, action_horizon, selected_dims)
        st.plotly_chart(fig, use_container_width=True)

    # Tab 2: Error heatmap
    with tabs[1]:
        fig = build_error_heatmap(gt_actions, pred_actions, action_keys, action_horizon)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### ⏱️ 误差随时间演变")
        fig2 = build_error_timeseries(gt_actions, pred_actions, action_keys, action_horizon)
        st.plotly_chart(fig2, use_container_width=True)

    # Tab 3: Per-dimension bar chart
    with tabs[2]:
        fig = build_per_dim_metrics_bar(gt_actions, pred_actions, action_keys)
        st.plotly_chart(fig, use_container_width=True)

        # Per-dimension statistics table
        st.markdown("#### 📋 逐维度统计")
        import pandas as pd
        rows = []
        for d in range(D):
            lb = action_keys[d] if d < len(action_keys) else f"dim_{d}"
            err = gt_actions[:, d] - pred_actions[:, d]
            rows.append({
                "维度": lb,
                "MSE": f"{np.mean(err**2):.6f}",
                "MAE": f"{np.mean(np.abs(err)):.6f}",
                "Max Error": f"{np.max(np.abs(err)):.6f}",
                "Mean Bias": f"{np.mean(err):+.6f}",
                "Std": f"{np.std(err):.6f}",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # Tab 4: 3D trajectory
    with tabs[3]:
        if D >= 3:
            col1, col2, col3 = st.columns(3)
            with col1:
                xd = st.selectbox("X 维度", all_dims, index=0, format_func=lambda d: action_keys[d], key="3d_x")
            with col2:
                yd = st.selectbox("Y 维度", all_dims, index=min(1, D - 1), format_func=lambda d: action_keys[d], key="3d_y")
            with col3:
                zd = st.selectbox("Z 维度", all_dims, index=min(2, D - 1), format_func=lambda d: action_keys[d], key="3d_z")
            fig = build_3d_trajectory(gt_actions, pred_actions, (xd, yd, zd), action_keys)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("动作维度不足 3，无法绘制 3D 轨迹。")

    # Tab 5: Error distribution
    with tabs[4]:
        fig = build_error_distribution(gt_actions, pred_actions, action_keys)
        st.plotly_chart(fig, use_container_width=True)


def _show_usage_guide():
    """Show usage instructions."""
    with st.expander("💡 如何生成评估结果", expanded=True):
        st.markdown("""
```python
from vlalab.eval import OpenLoopEvaluator
from vlalab.eval.adapters import GR00TAdapter

# 1. 创建 Policy 适配器
adapter = GR00TAdapter(policy)

# 2. 创建评估器
evaluator = OpenLoopEvaluator(
    policy=adapter,
    dataset_path="/path/to/dataset",
    dataset_format="lerobot",
    dataset_loader=loader,
)

# 3. 运行评估并保存 (同时生成图片 + npy + JSON)
results = evaluator.evaluate(
    traj_ids=[0, 1, 2],
    max_steps=200,
    save_plots_dir="outputs/eval/",
)

# 4. 保存结果 JSON
evaluator.evaluate_and_save("outputs/eval/results.json", ...)
```

保存后在左侧指定 `outputs/eval/` 目录即可加载交互式可视化。
        """)
