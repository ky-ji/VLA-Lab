"""
VLA-Lab Inference Run Viewer

Step-by-step replay of inference sessions with multi-camera support.
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import base64
import cv2
from typing import Optional, Dict, Any

import vlalab

# Setup matplotlib fonts
try:
    from vlalab.viz.mpl_fonts import setup_matplotlib_fonts
    setup_matplotlib_fonts(verbose=False)
except Exception:
    pass


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
    
    def plot_replay_frame(self, step_idx: int):
        """Plot replay frame for a step."""
        if not self.valid or step_idx >= len(self.steps):
            return
        
        step = self.steps[step_idx]
        current_state = self.get_step_state(step_idx)
        pred_action = self.get_step_action(step_idx)
        imgs = self.get_step_images(step_idx)
        timing = step.get("timing", {})
        
        # Layout
        c1, c2 = st.columns([1, 1.5])
        
        with c1:
            st.markdown("#### 👁️ 模型视觉观测")
            if imgs:
                # Display in a grid (up to 3 columns) to support multi-camera runs
                cam_names = list(imgs.keys())
                n_cols = min(3, max(1, len(cam_names)))
                cols = st.columns(n_cols)
                for i, cam in enumerate(cam_names):
                    with cols[i % n_cols]:
                        img = imgs[cam]
                        st.image(
                            img,
                            caption=f"{cam} | Step {step_idx} | {img.shape}",
                            use_container_width=True,
                        )
            else:
                st.warning("无图像数据")
            
            # Timing metrics
            if timing:
                t_transport = self._get_latency_ms(timing, "transport_latency")
                t_infer = self._get_latency_ms(timing, "inference_latency")
                total = self._get_latency_ms(timing, "total_latency")
                
                st.markdown("#### ⏱️ 时延诊断")
                col_t1, col_t2, col_t3 = st.columns(3)
                col_t1.metric("传输延迟", f"{t_transport:.0f} ms")
                col_t2.metric("推理耗时", f"{t_infer:.0f} ms")
                col_t3.metric("总回路", f"{total:.0f} ms")
                
                if t_transport > 100:
                    st.error(f"⚠️ 传输延迟过高 ({t_transport:.0f}ms)! 检查网络或 SSH 隧道")
        
        with c2:
            st.markdown("#### 🗺️ 3D 动作规划")
            
            if len(current_state) >= 3:
                fig = plt.figure(figsize=(8, 6))
                ax = fig.add_subplot(111, projection='3d')
                
                # Plot history
                all_states = self.get_all_states()
                if len(all_states) > 0 and all_states.shape[1] >= 3:
                    start = max(0, step_idx - 50)
                    hist = all_states[start:step_idx+1]
                    if len(hist) > 1:
                        ax.plot(hist[:,0], hist[:,1], hist[:,2], 'k-', alpha=0.3, label='History')
                
                # Current position
                ax.scatter(current_state[0], current_state[1], current_state[2], 
                          c='b', s=100, label='Current')
                
                # Predicted trajectory
                if len(pred_action) > 0 and pred_action.ndim == 2 and pred_action.shape[1] >= 3:
                    ax.plot(pred_action[:,0], pred_action[:,1], pred_action[:,2], 
                           'r--', linewidth=2, label='Pred')
                    ax.scatter(pred_action[-1,0], pred_action[-1,1], pred_action[-1,2], 
                              c='r', marker='x', s=100)
                
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Z')
                ax.legend()
                
                # Set axis limits
                if len(all_states) > 0 and all_states.shape[1] >= 3:
                    margin = 0.1
                    ax.set_xlim(all_states[:,0].min()-margin, all_states[:,0].max()+margin)
                    ax.set_ylim(all_states[:,1].min()-margin, all_states[:,1].max()+margin)
                    ax.set_zlim(all_states[:,2].min()-margin, all_states[:,2].max()+margin)
                
                st.pyplot(fig)
                plt.close(fig)
            else:
                st.warning("状态维度不足，无法绘制3D轨迹")
    
    def plot_latency_analysis(self):
        """Plot latency analysis chart."""
        if not self.steps:
            st.warning("当前日志不包含步骤数据")
            return
        
        steps_range = range(len(self.steps))
        trans_lats = []
        infer_lats = []
        total_lats = []
        
        for step in self.steps:
            timing = step.get("timing", {})
            trans_lats.append(self._get_latency_ms(timing, "transport_latency"))
            infer_lats.append(self._get_latency_ms(timing, "inference_latency"))
            total_lats.append(self._get_latency_ms(timing, "total_latency"))
        
        if not any(total_lats):
            st.warning("当前日志不包含详细时延数据")
            return
        
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(steps_range, total_lats, color='gray', alpha=0.3, label='Total Loop')
        ax.plot(steps_range, trans_lats, color='orange', label='Transport (Network)')
        ax.plot(steps_range, infer_lats, color='blue', label='Inference (GPU)')
        
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
        
        if valid_total:
            col1.metric("平均总延迟", f"{np.mean(valid_total):.1f} ms")
        if valid_trans:
            col2.metric("平均传输延迟", f"{np.mean(valid_trans):.1f} ms")
        if valid_infer:
            col3.metric("平均推理延迟", f"{np.mean(valid_infer):.1f} ms")
        if valid_total:
            col4.metric("最大总延迟", f"{np.max(valid_total):.1f} ms")


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
    
    # Tabs
    tab1, tab2 = st.tabs(["📺 逐帧回放", "📈 性能分析"])
    
    with tab1:
        if viewer.steps:
            step_idx = st.slider("Step", 0, len(viewer.steps)-1, 0)
            viewer.plot_replay_frame(step_idx)
        else:
            st.warning("无步骤数据")
    
    with tab2:
        viewer.plot_latency_analysis()
