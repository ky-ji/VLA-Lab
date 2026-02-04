"""
VLA-Lab Dataset Viewer

Browse and analyze training/evaluation datasets in Zarr format.
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
from typing import Optional, List

# Setup matplotlib fonts
try:
    from vlalab.viz.mpl_fonts import setup_matplotlib_fonts
    setup_matplotlib_fonts(verbose=False)
except Exception:
    pass


class ZarrDatasetViewer:
    """Viewer for Zarr datasets (Diffusion Policy format)."""
    
    def __init__(self, zarr_path: str):
        self.zarr_path = zarr_path
        self.valid = False
        self._load_data()
    
    def _load_data(self):
        """Load Zarr dataset."""
        try:
            import zarr
            self.root = zarr.open(self.zarr_path, mode='r')
            self.data = self.root['data']
            self.meta = self.root['meta']
            self.episode_ends = self.meta['episode_ends'][:]
            self.valid = True
        except ImportError:
            st.error("请安装 zarr: pip install zarr")
            self.valid = False
        except Exception as e:
            st.error(f"无法加载 Zarr 文件: {e}")
            self.valid = False
            return
        
        # Analyze fields
        self.image_keys = []
        self.lowdim_keys = []
        self.action_key = 'action'
        
        for key in self.data.keys():
            arr = self.data[key]
            if key == 'action':
                self.action_key = key
            elif len(arr.shape) == 4:  # (T, H, W, C)
                self.image_keys.append(key)
            else:
                self.lowdim_keys.append(key)
    
    def get_episode_slice(self, episode_idx: int) -> slice:
        """Get slice for an episode."""
        start_idx = 0 if episode_idx == 0 else self.episode_ends[episode_idx - 1]
        end_idx = self.episode_ends[episode_idx]
        return slice(int(start_idx), int(end_idx))
    
    def get_episode_data(self, episode_idx: int) -> dict:
        """Get all data for an episode."""
        s = self.get_episode_slice(episode_idx)
        data = {}
        for key in self.data.keys():
            data[key] = self.data[key][s]
        return data
    
    def plot_images_grid(self, episode_idx: int, step_interval: int = 5, max_frames: int = 20):
        """Plot image grid for an episode."""
        if not self.image_keys:
            st.warning("数据集中没有图像数据")
            return
        
        episode_data = self.get_episode_data(episode_idx)
        image_key = self.image_keys[0]
        images = episode_data[image_key]
        
        total_frames = len(images)
        frame_indices = list(range(0, total_frames, step_interval))[:max_frames]
        n_frames = len(frame_indices)
        
        n_cols = 5
        n_rows = (n_frames + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(3*n_cols, 3*n_rows))
        st.write(f"**Episode {episode_idx} - 图像概览 ({image_key})**")
        
        axes = np.atleast_2d(axes)
        
        for i, frame_idx in enumerate(frame_indices):
            row, col = i // n_cols, i % n_cols
            ax = axes[row, col]
            image = images[frame_idx]
            if image.shape[0] in [1, 3] and len(image.shape) == 3:
                image = np.transpose(image, (1, 2, 0))
            ax.imshow(image)
            ax.set_title(f'Step {frame_idx}', fontsize=8)
            ax.axis('off')
        
        for i in range(n_frames, n_rows * n_cols):
            row, col = i // n_cols, i % n_cols
            axes[row, col].axis('off')
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    
    def plot_actions_summary(self, episode_idx: int):
        """Plot action summary for an episode."""
        episode_data = self.get_episode_data(episode_idx)
        actions = episode_data['action']
        T = len(actions)
        action_dim = actions.shape[1]
        
        fig = plt.figure(figsize=(12, 8))
        gs = GridSpec(2, 2, figure=fig)
        
        st.write(f"**Episode {episode_idx} - 动作全局分析**")
        
        # Position time series
        ax1 = fig.add_subplot(gs[0, :])
        for i, label in enumerate(['x', 'y', 'z']):
            if i < action_dim:
                ax1.plot(actions[:, i], label=label, alpha=0.8)
        ax1.set_title('位置变化 (Position)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 3D trajectory
        ax2 = fig.add_subplot(gs[1, 0], projection='3d')
        if action_dim >= 3:
            ax2.plot(actions[:, 0], actions[:, 1], actions[:, 2], 'b-', alpha=0.6)
            ax2.scatter(actions[0, 0], actions[0, 1], actions[0, 2], c='g', label='Start')
            ax2.scatter(actions[-1, 0], actions[-1, 1], actions[-1, 2], c='r', label='End')
            ax2.set_title('3D 轨迹')
            ax2.set_xlabel('X')
            ax2.set_ylabel('Y')
            ax2.set_zlabel('Z')
        
        # Gripper
        ax3 = fig.add_subplot(gs[1, 1])
        if action_dim >= 8:
            ax3.plot(actions[:, 7], 'k-', linewidth=2)
            ax3.set_title('夹爪 (Gripper)')
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, "无夹爪数据", ha='center')
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    
    def plot_interactive_step(self, episode_idx: int, step_idx: int):
        """Plot interactive step view."""
        episode_data = self.get_episode_data(episode_idx)
        actions = episode_data['action']
        
        # Get image
        image = None
        if self.image_keys:
            image_key = self.image_keys[0]
            raw_img = episode_data[image_key][step_idx]
            if raw_img.shape[0] in [1, 3] and len(raw_img.shape) == 3:
                image = np.transpose(raw_img, (1, 2, 0))
            else:
                image = raw_img
        
        # Get action
        current_action = actions[step_idx]
        action_dim = actions.shape[1]
        
        # Layout
        c1, c2 = st.columns([1.5, 1])
        
        with c1:
            st.markdown(f"#### 📸 相机视角 (Step {step_idx})")
            if image is not None:
                st.image(image, use_container_width=True)
            else:
                st.warning("无图像数据")
        
        with c2:
            st.markdown("#### 🤖 机器人状态")
            st.info(f"Step: **{step_idx}** / {len(actions)-1}")
            
            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("X", f"{current_action[0]:.3f}")
            mc2.metric("Y", f"{current_action[1]:.3f}")
            mc3.metric("Z", f"{current_action[2]:.3f}")
            
            if action_dim >= 7:
                st.markdown("**姿态 (Quaternion):**")
                st.code(f"[{current_action[3]:.3f}, {current_action[4]:.3f}, {current_action[5]:.3f}, {current_action[6]:.3f}]")
            
            if action_dim >= 8:
                g_val = current_action[7]
                g_state = "OPEN" if g_val > 0.5 else "CLOSED"
                st.metric("夹爪 Gripper", f"{g_val:.3f}", delta=g_state)
        
        st.divider()
        
        # Trajectory view
        st.markdown("#### 📍 轨迹同步视图")
        
        fig = plt.figure(figsize=(14, 5))
        gs = GridSpec(1, 2, figure=fig)
        
        # 3D trajectory
        ax1 = fig.add_subplot(gs[0, 0], projection='3d')
        ax1.plot(actions[:,0], actions[:,1], actions[:,2], 'b-', alpha=0.2, linewidth=1, label='Path')
        ax1.scatter(actions[0,0], actions[0,1], actions[0,2], c='g', s=20, alpha=0.5)
        ax1.scatter(actions[-1,0], actions[-1,1], actions[-1,2], c='gray', s=20, alpha=0.5)
        ax1.scatter(current_action[0], current_action[1], current_action[2], 
                   c='r', s=150, edgecolors='k', label='Current', zorder=100)
        ax1.set_title("3D 空间位置 (红点=当前)", fontsize=10)
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        
        # Time series
        ax2 = fig.add_subplot(gs[0, 1])
        steps = np.arange(len(actions))
        for i, label in enumerate(['x', 'y', 'z']):
            ax2.plot(steps, actions[:, i], label=label, alpha=0.6)
        ax2.axvline(x=step_idx, color='r', linestyle='--', linewidth=2, label='Current Step')
        ax2.set_title("XYZ 随时间变化 (红线=当前)", fontsize=10)
        ax2.legend(loc='upper right', fontsize=8)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    
    def plot_workspace_3d(self, sample_ratio: float = 0.1):
        """Plot 3D workspace distribution."""
        if not self.valid:
            return
        
        st.write("正在采样并生成 3D 工作空间...")
        actions = self.data['action'][:]
        n_samples = int(len(actions) * sample_ratio)
        indices = np.random.choice(len(actions), n_samples, replace=False)
        sampled = actions[indices]
        
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection='3d')
        img = ax.scatter(sampled[:, 0], sampled[:, 1], sampled[:, 2],
                        c=np.arange(n_samples), cmap='viridis', s=1, alpha=0.3)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Global Workspace ({n_samples} points)')
        plt.colorbar(img, ax=ax, label='Time step order')
        st.pyplot(fig)
        plt.close(fig)


def render():
    """Render the dataset viewer page."""
    st.title("📊 训练数据可视化")
    
    # Path input
    default_path = st.sidebar.text_input(
        "Zarr 数据集路径",
        value="/data0/vla-data/processed/Diffusion_Policy/data/001_assembly_chocolate/assembly_chocolate_300.zarr"
    )
    
    # Initialize session state
    if 'zarr_viz' not in st.session_state:
        st.session_state.zarr_viz = None
    
    # Load button
    if st.sidebar.button("加载/重载数据集", type="primary"):
        if Path(default_path).exists():
            st.session_state.zarr_viz = ZarrDatasetViewer(default_path)
            st.success(f"已加载: {Path(default_path).name}")
        else:
            st.error("路径不存在！")
    
    viz = st.session_state.zarr_viz
    
    if viz and viz.valid:
        st.sidebar.markdown("---")
        st.sidebar.info(f"Episodes: {len(viz.episode_ends)}\nTotal Steps: {viz.episode_ends[-1]}")
        
        # Episode selection
        selected_ep = st.sidebar.selectbox("选择 Episode", range(len(viz.episode_ends)))
        
        # Tabs
        tab1, tab2, tab3 = st.tabs(["🔍 详细交互", "📊 全局概览", "🧊 空间分布"])
        
        with tab1:
            st.markdown(f"### Episode {selected_ep} - 逐帧分析")
            ep_data = viz.get_episode_data(selected_ep)
            max_step = len(ep_data['action']) - 1
            step_idx = st.slider("⏱️ 时间轴", 0, max_step, 0)
            viz.plot_interactive_step(selected_ep, step_idx)
        
        with tab2:
            col1, col2 = st.columns([1, 4])
            with col1:
                st.caption("设置")
                interval = st.slider("采样间隔", 1, 20, 5)
                max_frames = st.slider("最大帧数", 5, 50, 20)
            with col2:
                viz.plot_images_grid(selected_ep, step_interval=interval, max_frames=max_frames)
                st.divider()
                viz.plot_actions_summary(selected_ep)
        
        with tab3:
            ratio = st.slider("采样比例", 0.01, 1.0, 0.1, 0.01)
            if st.button("生成 3D 分布图"):
                viz.plot_workspace_3d(ratio)
    else:
        st.info("👈 请先在左侧输入路径并点击"加载数据集"")
