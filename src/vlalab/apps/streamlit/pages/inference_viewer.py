"""
VLA-Lab Inference Run Viewer

Step-by-step replay of inference sessions with multi-camera support.
Enhanced with detailed model inference information and interactive plots.
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import json
import cv2
import shutil
from typing import Optional, Dict, Any, List, Tuple

import vlalab

# Setup matplotlib fonts
try:
    from vlalab.viz.mpl_fonts import setup_matplotlib_fonts
    setup_matplotlib_fonts(verbose=False)
except Exception:
    pass


# Action 维度标签 (通用)
ACTION_DIM_LABELS = {
    8: ["x", "y", "z", "qx", "qy", "qz", "qw", "gripper"],
    7: ["x", "y", "z", "qx", "qy", "qz", "qw"],
    6: ["x", "y", "z", "rx", "ry", "rz"],
    3: ["x", "y", "z"],
}


class InferenceRunViewer:
    """Viewer for VLA-Lab inference runs."""
    
    def __init__(self, run_path: str):
        self.run_path = Path(run_path)
        self.valid = False
        self.steps = []
        self.meta = {}
        self._load_data()
    
    def _load_data(self):
        """Load run data."""
        try:
            meta_path = self.run_path / "meta.json"
            steps_path = self.run_path / "steps.jsonl"
            
            if meta_path.exists():
                with open(meta_path, "r") as f:
                    self.meta = json.load(f)
            
            if steps_path.exists():
                self.steps = []
                with open(steps_path, "r") as f:
                    for line in f:
                        if line.strip():
                            self.steps.append(json.loads(line))
            
            self.valid = True
        except Exception as e:
            st.error(f"加载失败: {e}")
            self.valid = False
    
    def _get_latency_ms(self, timing_dict: Dict, key_base: str) -> float:
        """Get latency value in ms."""
        new_key = f"{key_base}_ms"
        if new_key in timing_dict and timing_dict[new_key] is not None:
            return timing_dict[new_key]
        if key_base in timing_dict and timing_dict[key_base] is not None:
            return timing_dict[key_base] * 1000
        return 0.0
    
    def _get_action_dim_labels(self, action_dim: int) -> List[str]:
        """Get labels for action dimensions."""
        if action_dim in ACTION_DIM_LABELS:
            return ACTION_DIM_LABELS[action_dim]
        return [f"dim_{i}" for i in range(action_dim)]

    def _get_state_dim_labels(self, state_dim: int) -> List[str]:
        """
        Get labels for state dimensions.

        Priority:
        1) meta["state_keys"] if present
        2) meta["extra"]["state_keys"] if present
        3) fall back to ACTION_DIM_LABELS mapping (common robotics convention)
        4) dim_i
        """
        if state_dim <= 0:
            return []

        # Allow callers to stash state dim labels in meta / meta.extra
        meta_state_keys = self.meta.get("state_keys")
        if isinstance(meta_state_keys, list) and all(isinstance(x, str) for x in meta_state_keys):
            if len(meta_state_keys) >= state_dim:
                return meta_state_keys[:state_dim]

        extra = self.meta.get("extra", {})
        if isinstance(extra, dict):
            extra_state_keys = extra.get("state_keys")
            if isinstance(extra_state_keys, list) and all(isinstance(x, str) for x in extra_state_keys):
                if len(extra_state_keys) >= state_dim:
                    return extra_state_keys[:state_dim]

        # Common fallback (pose/gripper-like)
        if state_dim in ACTION_DIM_LABELS:
            return ACTION_DIM_LABELS[state_dim]

        return [f"dim_{i}" for i in range(state_dim)]

    def _split_xyz_quat_gripper(self, dim_labels: List[str]) -> List[Tuple[str, str, List[int]]]:
        """
        Split dimensions into 3 semantic groups: xyz / quaternion(or rotation) / gripper.

        Returns a list of (group_key, group_title, indices).
        Always returns 3 groups (some may be empty).
        """
        n = len(dim_labels)
        labels_l = [str(x).lower() for x in dim_labels]

        def _find_indices(targets: List[str]) -> List[int]:
            idxs = [i for i, lb in enumerate(labels_l) if lb in targets]
            return idxs

        xyz = _find_indices(["x", "y", "z"])
        quat = _find_indices(["qx", "qy", "qz", "qw"])
        rot3 = _find_indices(["rx", "ry", "rz", "roll", "pitch", "yaw"])
        grip = _find_indices(["gripper", "grasp", "grip"])

        # Fallback by convention if labels don't match anything useful
        if not xyz and n >= 3:
            xyz = [0, 1, 2]
        if not quat:
            if rot3:
                quat = rot3
            elif n >= 7:
                quat = [3, 4, 5, 6]
        if not grip and n >= 8:
            grip = [7]

        # Deduplicate & keep in-bounds
        def _uniq_in_bounds(idxs: List[int]) -> List[int]:
            seen = set()
            out = []
            for i in idxs:
                if 0 <= i < n and i not in seen:
                    seen.add(i)
                    out.append(i)
            return out

        xyz = _uniq_in_bounds(xyz)
        quat = _uniq_in_bounds(quat)
        grip = _uniq_in_bounds(grip)

        # Avoid overlap if label-based search produced duplicates
        used = set(xyz)
        quat = [i for i in quat if i not in used]
        used.update(quat)
        grip = [i for i in grip if i not in used]

        # Titles: prefer showing q* when it is 4-dim, otherwise rotation
        quat_title = "姿态 (qx,qy,qz,qw)" if len(quat) == 4 else "旋转 (rx,ry,rz)"
        return [
            ("xyz", "位置 (x,y,z)", xyz),
            ("quat", quat_title, quat),
            ("gripper", "夹爪 (gripper)", grip),
        ]
    
    def load_image_from_ref(self, image_ref: Dict) -> Optional[np.ndarray]:
        """Load image from image reference."""
        if not image_ref:
            return None
        
        image_path = self.run_path / image_ref.get("path", "")
        if not image_path.exists():
            return None
        
        img = cv2.imread(str(image_path))
        if img is None:
            return None
        
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    def get_step_image(self, step_idx: int) -> Optional[np.ndarray]:
        """Get image for a step."""
        if step_idx >= len(self.steps):
            return None
        
        step = self.steps[step_idx]
        obs = step.get("obs", {})
        images = obs.get("images", [])
        if images:
            return self.load_image_from_ref(images[0])
        return None

    def get_step_images(self, step_idx: int) -> Dict[str, np.ndarray]:
        """Get all camera images for a step as {camera_name: image_rgb}."""
        if step_idx >= len(self.steps):
            return {}
        step = self.steps[step_idx]
        obs = step.get("obs", {})
        images = obs.get("images", [])
        out: Dict[str, np.ndarray] = {}
        for ref in images or []:
            if not isinstance(ref, dict):
                continue
            cam = ref.get("camera_name", "default")
            img = self.load_image_from_ref(ref)
            if img is not None:
                out[str(cam)] = img
        return out
    
    def get_step_state(self, step_idx: int) -> np.ndarray:
        """Get state for a step."""
        if step_idx >= len(self.steps):
            return np.array([])
        
        step = self.steps[step_idx]
        obs = step.get("obs", {})
        state = obs.get("state", [])
        return np.array(state) if state else np.array([])
    
    def get_step_action(self, step_idx: int) -> np.ndarray:
        """Get action for a step."""
        if step_idx >= len(self.steps):
            return np.array([])
        
        step = self.steps[step_idx]
        action_data = step.get("action", {})
        values = action_data.get("values", [])
        return np.array(values) if values else np.array([])
    
    def get_all_states(self) -> np.ndarray:
        """Get all states as array."""
        states = [self.get_step_state(i) for i in range(len(self.steps))]
        valid_states = [s for s in states if len(s) > 0]
        return np.array(valid_states) if valid_states else np.array([])
    
    def plot_action_chunk_visualization(self, pred_action: np.ndarray, step_idx: int):
        """Plot detailed action chunk visualization."""
        if len(pred_action) == 0:
            st.warning("无动作数据")
            return
        
        # Ensure 2D array (chunk_size, action_dim)
        if pred_action.ndim == 1:
            pred_action = pred_action.reshape(1, -1)
        
        chunk_size, action_dim = pred_action.shape
        dim_labels = self._get_action_dim_labels(action_dim)
        
        # Create figure with two subplots
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Left: Action chunk heatmap
        ax1 = axes[0]
        im = ax1.imshow(pred_action.T, aspect='auto', cmap='RdBu_r', 
                        vmin=-1, vmax=1)
        ax1.set_xlabel('Chunk Step')
        ax1.set_ylabel('Action Dimension')
        ax1.set_yticks(range(action_dim))
        ax1.set_yticklabels(dim_labels)
        ax1.set_xticks(range(chunk_size))
        ax1.set_title(f'Action Chunk (Step {step_idx})')
        plt.colorbar(im, ax=ax1, label='Value')
        
        # Right: First action bar chart
        ax2 = axes[1]
        first_action = pred_action[0]
        colors = ['#4CAF50' if v >= 0 else '#F44336' for v in first_action]
        bars = ax2.bar(range(action_dim), first_action, color=colors, alpha=0.8)
        ax2.set_xticks(range(action_dim))
        ax2.set_xticklabels(dim_labels, rotation=45, ha='right')
        ax2.set_ylabel('Value')
        ax2.set_title('First Action (Executed)')
        ax2.axhline(0, color='gray', linestyle='-', linewidth=0.5)
        ax2.set_ylim(-1.2, 1.2)
        
        # Add value labels on bars
        for bar, val in zip(bars, first_action):
            height = bar.get_height()
            ax2.annotate(f'{val:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3 if height >= 0 else -10),
                        textcoords="offset points",
                        ha='center', va='bottom' if height >= 0 else 'top',
                        fontsize=8)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    
    def plot_timing_timeline(self, timing: Dict, step_idx: int):
        """Plot detailed timing timeline."""
        # Extract timestamps
        client_send = timing.get("client_send")
        server_recv = timing.get("server_recv")
        infer_start = timing.get("infer_start")
        infer_end = timing.get("infer_end")
        send_timestamp = timing.get("send_timestamp")
        
        # Check if we have enough timestamp data
        timestamps = [t for t in [client_send, server_recv, infer_start, infer_end, send_timestamp] if t is not None]
        if len(timestamps) < 2:
            return  # Not enough data for timeline
        
        # Normalize to relative time (ms from client_send)
        base_time = client_send if client_send else min(timestamps)
        
        events = []
        if client_send:
            events.append(("Client Send", (client_send - base_time) * 1000, '#2196F3'))
        if server_recv:
            events.append(("Server Recv", (server_recv - base_time) * 1000, '#4CAF50'))
        if infer_start:
            events.append(("Infer Start", (infer_start - base_time) * 1000, '#FF9800'))
        if infer_end:
            events.append(("Infer End", (infer_end - base_time) * 1000, '#F44336'))
        if send_timestamp:
            events.append(("Action Sent", (send_timestamp - base_time) * 1000, '#9C27B0'))
        
        if len(events) < 2:
            return
        
        # Create timeline figure
        fig, ax = plt.subplots(figsize=(10, 2))
        
        # Plot events as points
        times = [e[1] for e in events]
        labels = [e[0] for e in events]
        colors = [e[2] for e in events]
        
        # Draw timeline
        ax.hlines(0, min(times), max(times), color='gray', linewidth=2)
        
        # Draw events
        for i, (label, t, color) in enumerate(events):
            ax.scatter(t, 0, s=200, c=color, zorder=5, marker='o')
            ax.annotate(f'{label}\n{t:.1f}ms', 
                       xy=(t, 0), 
                       xytext=(0, 20 if i % 2 == 0 else -30),
                       textcoords='offset points',
                       ha='center', va='bottom' if i % 2 == 0 else 'top',
                       fontsize=9,
                       bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.3))
        
        ax.set_xlim(min(times) - 5, max(times) + 5)
        ax.set_ylim(-1, 1)
        ax.axis('off')
        ax.set_title(f'Step {step_idx} 时间线', fontsize=10)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    
    def _extract_state_action_data(self) -> Tuple[np.ndarray, np.ndarray, List[str], List[str], int, int]:
        """
        Extract state and action data from all steps.
        
        Returns:
            states_arr: (step_count, max_state_dim) array
            actions_arr: (step_count, max_action_dim) array (first action of each chunk)
            state_dim_labels: labels for state dimensions
            action_dim_labels: labels for action dimensions
            max_state_dim: max state dimension
            max_action_dim: max action dimension
        """
        step_count = len(self.steps)
        states_raw = []
        actions_raw = []
        max_state_dim = 0
        max_action_dim = 0

        for step in self.steps:
            obs = step.get("obs", {})
            state = obs.get("state", [])
            if state:
                max_state_dim = max(max_state_dim, len(state))
            states_raw.append(state)

            action_values = step.get("action", {}).get("values", [])
            if action_values:
                first_action = action_values[0]
                max_action_dim = max(max_action_dim, len(first_action))
                actions_raw.append(first_action)
            else:
                actions_raw.append([])

        states_arr = np.full((step_count, max_state_dim), np.nan) if max_state_dim > 0 else np.array([])
        actions_arr = np.full((step_count, max_action_dim), np.nan) if max_action_dim > 0 else np.array([])

        for i, state in enumerate(states_raw):
            if state and max_state_dim > 0:
                states_arr[i, :len(state)] = state

        for i, action in enumerate(actions_raw):
            if action and max_action_dim > 0:
                actions_arr[i, :len(action)] = action

        state_dim_labels = self._get_state_dim_labels(max_state_dim) if max_state_dim > 0 else []
        action_dim_labels = self._get_action_dim_labels(max_action_dim) if max_action_dim > 0 else []

        return states_arr, actions_arr, state_dim_labels, action_dim_labels, max_state_dim, max_action_dim

    def _extract_expanded_execution_data(self) -> Tuple[np.ndarray, List[Tuple[int, int]], List[str], int]:
        """
        Expand all action chunks into a continuous execution timeline.
        
        Returns:
            expanded_actions: (total_exec_steps, action_dim) array
            chunk_boundaries: List of (start_exec_idx, end_exec_idx) for each inference step
            action_dim_labels: labels for action dimensions
            max_action_dim: max action dimension
        """
        expanded_actions_list = []
        chunk_boundaries = []
        max_action_dim = 0
        exec_idx = 0

        for step in self.steps:
            action_values = step.get("action", {}).get("values", [])
            if action_values:
                chunk = np.array(action_values)
                if chunk.ndim == 1:
                    chunk = chunk.reshape(1, -1)
                max_action_dim = max(max_action_dim, chunk.shape[1])
                chunk_start = exec_idx
                for action in chunk:
                    expanded_actions_list.append(action)
                    exec_idx += 1
                chunk_boundaries.append((chunk_start, exec_idx))
            else:
                # No action for this step, record empty boundary
                chunk_boundaries.append((exec_idx, exec_idx))

        if not expanded_actions_list:
            return np.array([]), chunk_boundaries, [], 0

        # Pad to uniform dimension
        expanded_actions = np.full((len(expanded_actions_list), max_action_dim), np.nan)
        for i, action in enumerate(expanded_actions_list):
            expanded_actions[i, :len(action)] = action

        action_dim_labels = self._get_action_dim_labels(max_action_dim)
        return expanded_actions, chunk_boundaries, action_dim_labels, max_action_dim

    def plot_global_state_action(self, step_idx: int):
        """Plot global state/action trends and highlight current action chunk (Interactive).
        
        This is the INFERENCE step view - X axis is inference call index.
        """
        if not self.steps:
            st.warning("无步骤数据")
            return

        step_count = len(self.steps)
        states_raw = []
        actions_raw = []
        max_state_dim = 0
        max_action_dim = 0

        # --- 数据提取 ---
        for step in self.steps:
            obs = step.get("obs", {})
            state = obs.get("state", [])
            if state:
                max_state_dim = max(max_state_dim, len(state))
            states_raw.append(state)

            action_values = step.get("action", {}).get("values", [])
            if action_values:
                first_action = action_values[0]
                max_action_dim = max(max_action_dim, len(first_action))
                actions_raw.append(first_action)
            else:
                actions_raw.append([])

        if max_state_dim == 0 and max_action_dim == 0:
            st.warning("无状态与动作数据")
            return

        states_arr = np.full((step_count, max_state_dim), np.nan) if max_state_dim > 0 else None
        actions_arr = np.full((step_count, max_action_dim), np.nan) if max_action_dim > 0 else None

        for i, state in enumerate(states_raw):
            if state:
                states_arr[i, :len(state)] = state

        for i, action in enumerate(actions_raw):
            if action:
                actions_arr[i, :len(action)] = action

        # 按实际维度完整展示，不做截断
        state_dims = max_state_dim if max_state_dim > 0 else 0
        action_dims = max_action_dim if max_action_dim > 0 else 0
        state_dim_labels = self._get_state_dim_labels(max_state_dim) if max_state_dim > 0 else []
        action_dim_labels = self._get_action_dim_labels(max_action_dim) if max_action_dim > 0 else []

        # --- 按 xyz / (q or r) / gripper 分 3 行；State/Action 分 2 列 ---
        state_groups = self._split_xyz_quat_gripper(state_dim_labels[:state_dims] if state_dims > 0 else [])
        action_groups = self._split_xyz_quat_gripper(action_dim_labels[:action_dims] if action_dims > 0 else [])

        subplot_titles = []
        for i in range(3):
            subplot_titles.append(f"状态｜{state_groups[i][1]}")
            subplot_titles.append(f"动作｜{action_groups[i][1]}")

        fig = make_subplots(
            rows=3, cols=2,
            shared_xaxes=True,
            vertical_spacing=0.08,
            horizontal_spacing=0.06,
            subplot_titles=subplot_titles,
            row_heights=[0.33, 0.33, 0.34],
        )

        # 1) State traces
        state_group_title_set = False
        for row_idx, (gk, _gt, idxs) in enumerate(state_groups, start=1):
            if not idxs or max_state_dim <= 0:
                fig.add_annotation(
                    text="无该维度",
                    showarrow=False,
                    x=0.5, y=0.5,
                    xref="x domain", yref="y domain",
                    row=row_idx, col=1,
                )
                continue
            for d in idxs:
                label = state_dim_labels[d] if d < len(state_dim_labels) else f"dim_{d}"
                fig.add_trace(
                    go.Scatter(
                        y=states_arr[:, d],
                        name=f"s{d}({label})",
                        mode="lines",
                        line=dict(width=1.5),
                        legendgroup="state",
                        legendgrouptitle_text="State",
                        legendrank=1,
                        hoverinfo="y+name",
                    ),
                    row=row_idx, col=1,
                )
                # 只给第一条 state trace 设置 group title，避免重复显示
                if not state_group_title_set:
                    state_group_title_set = True
                else:
                    # 后续 trace 不再重复 title
                    fig.data[-1].legendgrouptitle = None

        # 2) Action traces (per group row) - 推理步视图只显示执行的第一个动作
        action_group_title_set = False

        for row_idx, (gk, _gt, idxs) in enumerate(action_groups, start=1):
            if not idxs or max_action_dim <= 0:
                fig.add_annotation(
                    text="无该维度",
                    showarrow=False,
                    x=0.5, y=0.5,
                    xref="x domain", yref="y domain",
                    row=row_idx, col=2,
                )
                continue

            # 历史动作曲线（每次推理执行的第一个动作）
            for d in idxs:
                label = action_dim_labels[d] if d < len(action_dim_labels) else f"dim_{d}"
                fig.add_trace(
                    go.Scatter(
                        y=actions_arr[:, d],
                        name=f"a{d}({label})",
                        mode="lines",
                        line=dict(width=1.5),
                        legendgroup="action",
                        legendgrouptitle_text="Action",
                        legendrank=2,
                        hoverinfo="y+name",
                    ),
                    row=row_idx, col=2,
                )
                if not action_group_title_set:
                    action_group_title_set = True
                else:
                    fig.data[-1].legendgrouptitle = None

        # 3) 当前步指示线：覆盖 3x2 所有子图
        for r in range(1, 4):
            for c in range(1, 3):
                fig.add_vline(
                    x=step_idx,
                    line_width=1,
                    line_dash="dash",
                    line_color="black",
                    row=r, col=c,
                )

        # 4) 更新布局
        fig.update_layout(
            height=780,
            hovermode="x unified",
            margin=dict(l=20, r=20, t=40, b=110),
            legend=dict(
                orientation="h",  # 横向，多行自动换行
                x=0.0,
                y=-0.18,
                xanchor="left",
                yanchor="top",
                bgcolor="rgba(255,255,255,0.65)",
                bordercolor="rgba(0,0,0,0.10)",
                borderwidth=1,
                font=dict(size=11),
                itemsizing="constant",
                itemclick="toggle",
                itemdoubleclick="toggleothers",
                entrywidth=95,
                entrywidthmode="pixels",
            ),
            template="plotly_white",
        )

        st.plotly_chart(fig, use_container_width=True)

    def plot_global_execution_view(self, step_idx: int):
        """
        Plot EXECUTION step view - X axis is global execution step index.
        
        This expands all action chunks into a continuous timeline, showing
        what actions would be executed if each chunk was fully consumed.
        """
        if not self.steps:
            st.warning("无步骤数据")
            return

        # Extract expanded execution data
        expanded_actions, chunk_boundaries, action_dim_labels, max_action_dim = self._extract_expanded_execution_data()
        
        if len(expanded_actions) == 0 or max_action_dim == 0:
            st.warning("无动作数据可展开")
            return

        total_exec_steps = len(expanded_actions)
        action_groups = self._split_xyz_quat_gripper(action_dim_labels[:max_action_dim])

        # Get current chunk's execution range
        current_chunk_start, current_chunk_end = chunk_boundaries[step_idx] if step_idx < len(chunk_boundaries) else (0, 0)
        current_chunk_len = current_chunk_end - current_chunk_start

        # Build subplot titles
        subplot_titles = [f"执行步动作｜{action_groups[i][1]}" for i in range(3)]

        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            subplot_titles=subplot_titles,
            row_heights=[0.33, 0.33, 0.34],
        )

        # Get current chunk's predicted actions for overlay
        current_action_values = self.steps[step_idx].get("action", {}).get("values", [])
        pred_chunk = None
        pred_x_range = None
        if current_action_values:
            pred_chunk = np.array(current_action_values)
            if pred_chunk.ndim == 1:
                pred_chunk = pred_chunk.reshape(1, -1)
            # X range for predicted chunk in execution step space
            pred_x_range = np.arange(current_chunk_start, current_chunk_start + len(pred_chunk))

        # Plot action traces for each group
        action_group_title_set = False
        pred_group_title_set = False
        for row_idx, (gk, _gt, idxs) in enumerate(action_groups, start=1):
            if not idxs:
                fig.add_annotation(
                    text="无该维度",
                    showarrow=False,
                    x=0.5, y=0.5,
                    xref="x domain", yref="y domain",
                    row=row_idx, col=1,
                )
                continue

            # Full execution timeline
            for d in idxs:
                label = action_dim_labels[d] if d < len(action_dim_labels) else f"dim_{d}"
                fig.add_trace(
                    go.Scatter(
                        y=expanded_actions[:, d],
                        name=f"exec_a{d}({label})",
                        mode="lines",
                        line=dict(width=1.5),
                        legendgroup="exec_action",
                        legendgrouptitle_text="Execution Action",
                        legendrank=1,
                        hoverinfo="y+name",
                    ),
                    row=row_idx, col=1,
                )
                if not action_group_title_set:
                    action_group_title_set = True
                else:
                    fig.data[-1].legendgrouptitle = None

            # Highlight current chunk region
            if current_chunk_len > 0:
                chunk_label = f"Chunk {step_idx} ({current_chunk_len} steps)"
                fig.add_vrect(
                    x0=current_chunk_start, x1=current_chunk_end,
                    fillcolor="orange", opacity=0.2,
                    layer="below", line_width=0,
                    row=row_idx, col=1,
                    annotation_text=chunk_label if row_idx == 1 else None,
                    annotation_position="top left",
                )

            # Draw predicted chunk trajectory (overlay with different style)
            if pred_chunk is not None and pred_x_range is not None and len(pred_x_range) > 0:
                for d in idxs:
                    if d >= pred_chunk.shape[1]:
                        continue
                    label = action_dim_labels[d] if d < len(action_dim_labels) else f"dim_{d}"
                    fig.add_trace(
                        go.Scatter(
                            x=pred_x_range,
                            y=pred_chunk[:, d],
                            name=f"Pred a{d}({label})",
                            mode="lines+markers",
                            marker=dict(size=5, symbol="circle"),
                            line=dict(dash="dot", width=2, color="rgba(255, 100, 0, 0.8)"),
                            legendgroup="pred_action",
                            legendgrouptitle_text="Predicted Chunk",
                            legendrank=2,
                            hovertemplate=f"Pred a{d}({label}): %{{y:.3f}}<extra></extra>",
                        ),
                        row=row_idx, col=1,
                    )
                    if not pred_group_title_set:
                        pred_group_title_set = True
                    else:
                        fig.data[-1].legendgrouptitle = None

            # Add vertical lines at chunk boundaries (lighter color)
            for i, (start, end) in enumerate(chunk_boundaries):
                if start != end:  # Only draw for non-empty chunks
                    fig.add_vline(
                        x=start,
                        line_width=0.5,
                        line_dash="dot",
                        line_color="gray",
                        opacity=0.3,
                        row=row_idx, col=1,
                    )

        # Current execution position indicator (first action of current chunk)
        for r in range(1, 4):
            fig.add_vline(
                x=current_chunk_start,
                line_width=2,
                line_dash="dash",
                line_color="red",
                row=r, col=1,
            )

        # Update layout
        fig.update_layout(
            height=650,
            hovermode="x unified",
            margin=dict(l=20, r=20, t=40, b=80),
            legend=dict(
                orientation="h",
                x=0.0,
                y=-0.12,
                xanchor="left",
                yanchor="top",
                bgcolor="rgba(255,255,255,0.65)",
                bordercolor="rgba(0,0,0,0.10)",
                borderwidth=1,
                font=dict(size=11),
                itemsizing="constant",
            ),
            template="plotly_white",
        )

        # Add x-axis label
        fig.update_xaxes(title_text="全局执行步 (Execution Step)", row=3, col=1)

        st.plotly_chart(fig, use_container_width=True)

        # Show statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("总执行步数", total_exec_steps)
        with col2:
            st.metric("总推理次数", len(self.steps))
        with col3:
            avg_chunk = total_exec_steps / len(self.steps) if self.steps else 0
            st.metric("平均Chunk大小", f"{avg_chunk:.1f}")
        with col4:
            st.metric("当前Chunk位置", f"{current_chunk_start}-{current_chunk_end}")

    def render_inference_details_panel(self, step_idx: int, show_language_prompt: bool = True):
        """Render detailed inference information panel."""
        if step_idx >= len(self.steps):
            return
        
        step = self.steps[step_idx]
        prompt = step.get("prompt")
        tags = step.get("tags", {})
        
        # Language Prompt
        if show_language_prompt:
            if prompt:
                st.markdown("#### 🗣️ 语言指令")
                st.info(f"**Prompt:** {prompt}")
            elif self.meta.get("task_prompt"):
                st.markdown("#### 🗣️ 语言指令")
                st.info(f"**Task Prompt:** {self.meta.get('task_prompt')}")
        
        # Global trends (Now Interactive) - Two views: Inference vs Execution
        st.markdown("#### 📈 全局状态与动作趋势")
        
        view_tab1, view_tab2 = st.tabs(["🔬 推理步视图", "⚡ 执行步视图"])
        
        with view_tab1:
            st.caption("X轴: 推理调用次数 | 显示每次推理时的状态和执行的第一个动作")
            self.plot_global_state_action(step_idx)
        
        with view_tab2:
            st.caption("X轴: 全局执行步 | 展开所有Chunk，显示完整动作执行序列")
            self.plot_global_execution_view(step_idx)

        # Tags
        if tags:
            st.markdown("#### 🏷️ 标签")
            st.json(tags)
        
        # Raw Step JSON
        with st.expander("🔍 原始 Step JSON"):
            st.json(step)
    
    def render_model_config_panel(self):
        """Render model configuration panel from metadata."""
        if not self.meta:
            return
        
        st.markdown("### 🤖 模型配置")
        
        # Basic info
        col1, col2, col3 = st.columns(3)
        with col1:
            model_type = self.meta.get("model_type", "unknown")
            st.metric("模型类型", model_type)
        with col2:
            action_dim = self.meta.get("action_dim", "N/A")
            st.metric("Action Dim", action_dim)
        with col3:
            action_horizon = self.meta.get("action_horizon", "N/A")
            st.metric("Action Horizon", action_horizon)
        
        col4, col5, col6 = st.columns(3)
        with col4:
            inference_freq = self.meta.get("inference_freq")
            if inference_freq:
                st.metric("推理频率", f"{inference_freq} Hz")
        with col5:
            robot_type = self.meta.get("robot_type", self.meta.get("robot_name", "unknown"))
            st.metric("机器人", robot_type)
        with col6:
            task_name = self.meta.get("task_name", "unknown")
            st.metric("任务", task_name)
        
        # Model path
        model_path = self.meta.get("model_path")
        if model_path:
            st.markdown(f"**Model Path:** `{model_path}`")
        
        # Cameras
        cameras = self.meta.get("cameras", [])
        if cameras:
            st.markdown("**相机配置:**")
            cam_names = [c.get("name", c.get("camera_name", f"cam_{i}")) for i, c in enumerate(cameras)]
            st.text(f"Cameras: {', '.join(cam_names)}")
        
        # Server/Client config
        server_config = self.meta.get("server_config", {})
        client_config = self.meta.get("client_config", {})
        
        if server_config or client_config:
            with st.expander("部署配置详情"):
                if server_config:
                    st.markdown("**Server Config:**")
                    st.json(server_config)
                if client_config:
                    st.markdown("**Client Config:**")
                    st.json(client_config)
    
    
    def plot_replay_frame(self, step_idx: int, show_details: bool = True):
        """Plot replay frame for a step."""
        if not self.valid or step_idx >= len(self.steps):
            return
        
        step = self.steps[step_idx]
        current_state = self.get_step_state(step_idx)
        pred_action = self.get_step_action(step_idx)
        imgs = self.get_step_images(step_idx)
        
        # Layout: 左侧“视觉观测 + 动作规划”；右侧上提“状态/动作视图”
        left_col, right_col = st.columns([1.5, 1.0])
        
        with left_col:
            st.markdown("#### 👁️ 模型视觉观测")
            if imgs:
                cam_names = list(imgs.keys())
                n_cols = min(3, max(1, len(cam_names)))
                cols = st.columns(n_cols)
                for i, cam in enumerate(cam_names):
                    with cols[i % n_cols]:
                        img = imgs[cam]
                        st.image(
                            img,
                            caption=f"{cam}",
                            use_container_width=True,
                        )
            else:
                st.warning("无图像数据")

            st.markdown("#### 🗺️ 3D 动作规划")
            
            if len(current_state) >= 3:
                fig = go.Figure()

                # --- 修复部分开始 ---
                # 1. 历史轨迹 (History)
                all_states = self.get_all_states()
                if len(all_states) > 0 and all_states.shape[1] >= 3:
                    # 获取从开始到当前步的所有轨迹
                    hist = all_states[:step_idx+1]
                    
                    if len(hist) > 1:
                        fig.add_trace(go.Scatter3d(
                            x=hist[:, 0], y=hist[:, 1], z=hist[:, 2],
                            mode='lines',
                            # 【修改点】：颜色改为纯深灰(#555555)，不透明，宽度加粗到4
                            line=dict(color='#555555', width=4), 
                            name='History',
                            hoverinfo='skip' 
                        ))
                # --- 修复部分结束 ---

                # 2. 预测轨迹 (Pred)
                if len(pred_action) > 0 and pred_action.ndim == 2 and pred_action.shape[1] >= 3:
                    fig.add_trace(go.Scatter3d(
                        x=pred_action[:, 0], y=pred_action[:, 1], z=pred_action[:, 2],
                        mode='lines+markers',
                        line=dict(color='rgba(255, 50, 50, 0.8)', width=4, dash='dot'),
                        marker=dict(size=2, color='rgba(255, 50, 50, 0.8)'),
                        name='Prediction'
                    ))
                    # 目标点
                    fig.add_trace(go.Scatter3d(
                        x=[pred_action[-1, 0]], y=[pred_action[-1, 1]], z=[pred_action[-1, 2]],
                        mode='markers',
                        marker=dict(size=5, color='red', symbol='diamond'),
                        name='Target'
                    ))

                # 3. 当前位置 (Current)
                fig.add_trace(go.Scatter3d(
                    x=[current_state[0]], y=[current_state[1]], z=[current_state[2]],
                    mode='markers',
                    marker=dict(size=6, color='#2196F3', symbol='circle', line=dict(width=1, color='white')),
                    name='Current'
                ))

                fig.update_layout(
                    height=500,
                    margin=dict(l=0, r=0, b=0, t=10),
                    scene=dict(
                        xaxis_title='X',
                        yaxis_title='Y',
                        zaxis_title='Z',
                        aspectmode='data',
                    ),
                    legend=dict(
                        x=0, y=1,
                        bgcolor='rgba(255,255,255,0.6)',
                        itemsizing='constant'
                    ),
                    uirevision='constant_scene_view'
                )
                
                st.plotly_chart(fig, use_container_width=True)

            else:
                st.warning("状态维度不足，无法绘制3D轨迹")

            # 左下方：语言指令
            prompt = step.get("prompt")
            task_prompt = self.meta.get("task_prompt") if self.meta else None
            if prompt or task_prompt:
                st.divider()
                st.markdown("#### 🗣️ 语言指令")
                if prompt:
                    st.info(f"**Prompt:** {prompt}")
                else:
                    st.info(f"**Task Prompt:** {task_prompt}")
        
        with right_col:
            if show_details:
                self.render_inference_details_panel(step_idx, show_language_prompt=False)
            else:
                st.info("已关闭推理详情（侧边栏勾选“显示推理详情”可查看状态/动作视图）")
            
    def plot_latency_analysis(self):
        """Plot latency analysis chart."""
        if not self.steps:
            st.warning("当前日志不包含步骤数据")
            return
        
        steps_range = range(len(self.steps))
        trans_lats = []
        infer_lats = []
        total_lats = []
        msg_intervals = []
        
        for step in self.steps:
            timing = step.get("timing", {})
            trans_lats.append(self._get_latency_ms(timing, "transport_latency"))
            infer_lats.append(self._get_latency_ms(timing, "inference_latency"))
            total_lats.append(self._get_latency_ms(timing, "total_latency"))
            msg_intervals.append(self._get_latency_ms(timing, "message_interval"))
        
        if not any(total_lats):
            st.warning("当前日志不包含详细时延数据")
            return
        
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(steps_range, total_lats, color='gray', alpha=0.3, label='Total Loop')
        ax.plot(steps_range, trans_lats, color='orange', label='Transport (Network)')
        ax.plot(steps_range, infer_lats, color='blue', label='Inference (GPU)')
        if any(msg_intervals):
            ax.plot(steps_range, msg_intervals, color='green', alpha=0.5, label='Message Interval')
        
        ax.set_title("时延组成分析 (ms)")
        ax.set_xlabel("Step")
        ax.set_ylabel("Latency (ms)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axhline(100, color='r', linestyle='--', alpha=0.5)
        ax.text(0, 105, '100ms Alert', color='r', fontsize=8)
        
        st.pyplot(fig)
        plt.close(fig)
        
        # Statistics
        col1, col2, col3, col4 = st.columns(4)
        valid_total = [t for t in total_lats if t > 0]
        valid_trans = [t for t in trans_lats if t > 0]
        valid_infer = [t for t in infer_lats if t > 0]
        valid_interval = [t for t in msg_intervals if t > 0]
        
        if valid_total:
            col1.metric("平均总延迟", f"{np.mean(valid_total):.1f} ms")
        if valid_trans:
            col2.metric("平均传输延迟", f"{np.mean(valid_trans):.1f} ms")
        if valid_infer:
            col3.metric("平均推理延迟", f"{np.mean(valid_infer):.1f} ms")
        if valid_total:
            col4.metric("最大总延迟", f"{np.max(valid_total):.1f} ms")
        
        # Additional statistics row
        if valid_interval or valid_infer:
            st.markdown("#### 📊 详细统计")
            col5, col6, col7, col8 = st.columns(4)
            
            if valid_infer:
                with col5:
                    st.metric("最小推理延迟", f"{np.min(valid_infer):.1f} ms")
                with col6:
                    st.metric("推理延迟 P95", f"{np.percentile(valid_infer, 95):.1f} ms")
            
            if valid_interval:
                with col7:
                    st.metric("平均帧间隔", f"{np.mean(valid_interval):.1f} ms")
                with col8:
                    freq = 1000.0 / np.mean(valid_interval) if np.mean(valid_interval) > 0 else 0
                    st.metric("实际频率", f"{freq:.1f} Hz")
    
    def plot_action_statistics(self):
        """Plot action statistics across all steps."""
        if not self.steps:
            st.warning("当前日志不包含步骤数据")
            return
        
        # Collect all first actions
        actions = []
        for step in self.steps:
            action_data = step.get("action", {})
            values = action_data.get("values", [])
            if values and len(values) > 0:
                actions.append(values[0])  # First action of each chunk
        
        if not actions:
            st.warning("无动作数据")
            return
        
        actions = np.array(actions)
        action_dim = actions.shape[1]
        dim_labels = self._get_action_dim_labels(action_dim)
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(14, 8))
        
        # 1. Action trajectory over time (position)
        ax1 = axes[0, 0]
        if action_dim >= 3:
            ax1.plot(actions[:, 0], label='x', alpha=0.8)
            ax1.plot(actions[:, 1], label='y', alpha=0.8)
            ax1.plot(actions[:, 2], label='z', alpha=0.8)
            ax1.set_xlabel('Step')
            ax1.set_ylabel('Position')
            ax1.set_title('Position over Time')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # 2. Action distribution (histogram)
        ax2 = axes[0, 1]
        ax2.boxplot([actions[:, i] for i in range(min(action_dim, 8))], labels=dim_labels[:min(action_dim, 8)])
        ax2.set_xlabel('Dimension')
        ax2.set_ylabel('Value')
        ax2.set_title('Action Distribution per Dimension')
        ax2.grid(True, alpha=0.3)
        
        # 3. Gripper state if available
        ax3 = axes[1, 0]
        if action_dim >= 8:
            ax3.plot(actions[:, 7], 'g-', label='Gripper', linewidth=1.5)
            ax3.fill_between(range(len(actions)), 0, actions[:, 7], alpha=0.3, color='green')
            ax3.set_xlabel('Step')
            ax3.set_ylabel('Gripper State')
            ax3.set_title('Gripper State over Time')
            ax3.set_ylim(-0.1, 1.1)
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'No Gripper Data', ha='center', va='center', fontsize=12)
            ax3.axis('off')
        
        # 4. Action magnitude
        ax4 = axes[1, 1]
        if action_dim >= 3:
            magnitudes = np.linalg.norm(actions[:, :3], axis=1)
            ax4.plot(magnitudes, 'b-', alpha=0.8)
            ax4.fill_between(range(len(magnitudes)), 0, magnitudes, alpha=0.2)
            ax4.set_xlabel('Step')
            ax4.set_ylabel('Magnitude')
            ax4.set_title('Position Action Magnitude')
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        
        # Statistics table
        st.markdown("#### 📋 动作统计")
        stats_data = {
            "Dimension": dim_labels[:min(action_dim, 8)],
            "Mean": [f"{np.mean(actions[:, i]):.4f}" for i in range(min(action_dim, 8))],
            "Std": [f"{np.std(actions[:, i]):.4f}" for i in range(min(action_dim, 8))],
            "Min": [f"{np.min(actions[:, i]):.4f}" for i in range(min(action_dim, 8))],
            "Max": [f"{np.max(actions[:, i]):.4f}" for i in range(min(action_dim, 8))],
        }
        st.dataframe(stats_data, use_container_width=True)


def render():
    """Render the inference viewer page."""
    st.title("🔬 推理运行回放")
    
    # Sidebar: show current runs directory
    runs_dir = vlalab.get_runs_dir()
    st.sidebar.markdown("### 日志目录")
    st.sidebar.code(str(runs_dir))
    
    # List projects
    projects = vlalab.list_projects()
    
    if not projects:
        st.info(f"未找到任何项目。请先使用 `vlalab.init()` 创建运行记录。\n\n日志目录: `{runs_dir}`")
        st.markdown("""
        **提示**: 设置 `$VLALAB_DIR` 环境变量可更改日志存储位置。
        
        ```bash
        export VLALAB_DIR=/path/to/your/logs
        ```
        """)
        return
    
    # Project filter
    selected_project = st.sidebar.selectbox(
        "选择项目",
        ["全部"] + projects,
    )
    
    # List runs
    if selected_project == "全部":
        run_paths = vlalab.list_runs()
    else:
        run_paths = vlalab.list_runs(project=selected_project)
    
    if not run_paths:
        st.info("该项目下没有运行记录。")
        return
    
    # Select run
    selected_path = st.sidebar.selectbox(
        "选择运行",
        run_paths,
        format_func=lambda p: f"{p.name} ({p.parent.name})"
    )
    
    if selected_path is None:
        return
    
    # Load viewer
    if 'viewer' not in st.session_state or st.session_state.get('last_run') != str(selected_path):
        st.session_state.viewer = InferenceRunViewer(str(selected_path))
        st.session_state.last_run = str(selected_path)
    
    viewer = st.session_state.viewer
    
    if not viewer.valid:
        return
    
    # Show metadata
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 运行信息")
    st.sidebar.info(f"步数: {len(viewer.steps)}")
    if viewer.meta:
        model = viewer.meta.get("model_name", "unknown")
        if isinstance(model, str) and len(model) > 30:
            model = "..." + model[-30:]
        st.sidebar.info(f"模型: {model}")
        
        # Additional metadata display
        model_type = viewer.meta.get("model_type")
        if model_type:
            st.sidebar.info(f"类型: {model_type}")
        
        task_name = viewer.meta.get("task_name")
        if task_name and task_name != "unknown":
            st.sidebar.info(f"任务: {task_name}")
    
    # Display options
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 显示选项")
    show_details = st.sidebar.checkbox("显示推理详情", value=True, help="显示详细的模型推理信息面板")
    
    # Delete run section
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚠️ 危险操作")
    
    # Use session state for delete confirmation
    delete_key = f"delete_confirm_{selected_path}"
    if delete_key not in st.session_state:
        st.session_state[delete_key] = False
    
    if not st.session_state[delete_key]:
        if st.sidebar.button("🗑️ 删除此运行", key="delete_run_btn", type="secondary"):
            st.session_state[delete_key] = True
            st.rerun()
    else:
        st.sidebar.warning(f"确认删除 **{selected_path.name}**？此操作不可撤销！")
        col1, col2 = st.sidebar.columns(2)
        with col1:
            if st.button("✅ 确认", key="confirm_delete", type="primary"):
                try:
                    shutil.rmtree(selected_path)
                    st.session_state[delete_key] = False
                    # Clear viewer cache
                    if 'viewer' in st.session_state:
                        del st.session_state['viewer']
                    if 'last_run' in st.session_state:
                        del st.session_state['last_run']
                    st.success(f"已删除: {selected_path.name}")
                    st.rerun()
                except Exception as e:
                    st.error(f"删除失败: {e}")
        with col2:
            if st.button("❌ 取消", key="cancel_delete"):
                st.session_state[delete_key] = False
                st.rerun()
    
    # Tabs - expanded with more views
    tab1, tab2, tab3 = st.tabs(["📺 逐帧回放", "🎯 动作分析", "🤖 模型配置"])
    
    with tab1:
        if viewer.steps:
            step_idx = st.slider("Step", 0, len(viewer.steps)-1, 0)
            viewer.plot_replay_frame(step_idx, show_details=show_details)
        else:
            st.warning("无步骤数据")
    
    with tab2:
        viewer.plot_action_statistics()
    
    with tab3:
        viewer.render_model_config_panel()
        
        # Raw metadata
        with st.expander("🔍 原始 Meta JSON"):
            st.json(viewer.meta)