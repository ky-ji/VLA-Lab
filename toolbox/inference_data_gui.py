import streamlit as st
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
import sys
import base64
import cv2

# --- 路径配置 ---
sys.path.insert(0, str(Path(__file__).parent.parent))
try:
    from toolbox.mpl_fonts import setup_matplotlib_fonts
    setup_matplotlib_fonts(verbose=False)
except Exception:
    pass

class InferenceGUI:
    def __init__(self, log_path):
        self.log_path = log_path
        self.valid = False
        self.load_data()
    
    def _get_latency_ms(self, timing_dict, key_base):
        """
        安全地获取时延值（兼容新旧格式）
        
        Args:
            timing_dict: 时间信息字典
            key_base: 字段名基础（如 'transport_latency'）
        
        Returns:
            时延值（毫秒）
        """
        # 优先使用新版格式（已经是毫秒）
        new_key = f"{key_base}_ms"
        if new_key in timing_dict and timing_dict[new_key] is not None:
            return timing_dict[new_key]
        # 使用旧版格式（秒转毫秒）
        old_key = key_base
        if old_key in timing_dict and timing_dict[old_key] is not None:
            return timing_dict[old_key] * 1000
        return 0.0

    def load_data(self):
        try:
            with open(self.log_path, 'r') as f:
                self.log_data = json.load(f)
            
            self.steps = self.log_data.get('steps', [])
            if not self.steps:
                st.error("日志文件为空")
                return

            self.states = []
            self.actions = []
            self.images = [] # 存储 Base64 字符串
            self.timings = [] # 存储时间信息
            
            for step in self.steps:
                # State
                self.states.append(step.get('input', {}).get('state', []))
                # Action
                action_data = step.get('action', {})
                self.actions.append(action_data.get('values', []))
                # Image
                self.images.append(step.get('input', {}).get('image_base64', None))
                # Timing
                self.timings.append(step.get('timing', {}))
            
            self.states = np.array(self.states)
            self.valid = True
            
        except Exception as e:
            st.error(f"加载失败: {e}")
            self.valid = False

    def decode_image(self, b64_str):
        if not b64_str: return None
        try:
            img_data = base64.b64decode(b64_str)
            img_array = np.frombuffer(img_data, dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except:
            return None

    def plot_replay_frame(self, step_idx):
        if not self.valid: return

        # 获取数据
        current_state = self.states[step_idx]
        pred_traj = np.array(self.actions[step_idx])
        img = self.decode_image(self.images[step_idx])
        timing = self.timings[step_idx]

        # --- 布局设计 ---
        # 第一行：左侧图像，右侧3D轨迹
        c1, c2 = st.columns([1, 1.5])
        
        with c1:
            st.markdown("#### 👁️ 模型视觉观测")
            if img is not None:
                st.image(img, caption=f"Step {step_idx} Input (Size: {img.shape})", use_container_width=True)
            else:
                st.warning("无图像数据 (旧版日志?)")
            
            # 显示关键时延指标（兼容新旧格式）
            if timing:
                t_transport = self._get_latency_ms(timing, 'transport_latency')
                t_infer = self._get_latency_ms(timing, 'inference_latency')
                total = self._get_latency_ms(timing, 'total_latency')
                
                st.markdown("#### ⏱️ 时延诊断")
                col_t1, col_t2, col_t3 = st.columns(3)
                col_t1.metric("传输延迟", f"{t_transport:.0f} ms", help="客户端拍照 -> 服务器接收")
                col_t2.metric("推理耗时", f"{t_infer:.0f} ms", help="模型前向传播时间")
                col_t3.metric("总回路", f"{total:.0f} ms", help="拍照 -> 收到动作")
                
                if t_transport > 100:
                    st.error(f"⚠️ 传输延迟过高 ({t_transport:.0f}ms)! 检查网络或 SSH 隧道")
                
                # 显示详细时间戳（如果可用）
                if timing.get('client_send') is not None:
                    st.markdown("#### 📊 详细时间线")
                    with st.expander("展开查看时间戳详情"):
                        if timing.get('client_send'):
                            st.text(f"客户端发送: {timing.get('client_send', 'N/A')}")
                        if timing.get('server_recv'):
                            st.text(f"服务器接收: {timing.get('server_recv', 'N/A')}")
                        if timing.get('infer_start'):
                            st.text(f"推理开始: {timing.get('infer_start', 'N/A')}")
                        if timing.get('infer_end'):
                            st.text(f"推理结束: {timing.get('infer_end', 'N/A')}")
                        if timing.get('send_timestamp'):
                            st.text(f"发送时间: {timing.get('send_timestamp', 'N/A')}")
                        if timing.get('message_interval_ms') is not None:
                            st.text(f"消息间隔: {timing.get('message_interval_ms', 'N/A'):.1f} ms")

        with c2:
            st.markdown("#### 🗺️ 3D 动作规划")
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111, projection='3d')
            
            # 画历史轨迹 (最近50步)
            start = max(0, step_idx - 50)
            hist = self.states[start:step_idx+1]
            if len(hist) > 1:
                ax.plot(hist[:,0], hist[:,1], hist[:,2], 'k-', alpha=0.3, label='History')
            
            # 画当前点
            ax.scatter(current_state[0], current_state[1], current_state[2], c='b', s=100, label='Current')
            
            # 画预测
            if len(pred_traj) > 0:
                ax.plot(pred_traj[:,0], pred_traj[:,1], pred_traj[:,2], 'r--', linewidth=2, label='Pred')
                ax.scatter(pred_traj[-1,0], pred_traj[-1,1], pred_traj[-1,2], c='r', marker='x')

            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.legend()
            
            # 固定坐标轴防止抖动
            margin = 0.1
            ax.set_xlim(self.states[:,0].min()-margin, self.states[:,0].max()+margin)
            ax.set_ylim(self.states[:,1].min()-margin, self.states[:,1].max()+margin)
            ax.set_zlim(self.states[:,2].min()-margin, self.states[:,2].max()+margin)
            
            st.pyplot(fig)
            plt.close(fig)

    def plot_latency_analysis(self):
        if not self.timings:
            st.warning("当前日志不包含时延数据")
            return
        
        # 检查是否有新版或旧版格式的时延数据
        has_new_format = any('inference_latency_ms' in t for t in self.timings)
        has_old_format = any('inference_latency' in t and 'inference_latency_ms' not in t for t in self.timings)
        
        if not (has_new_format or has_old_format):
            st.warning("当前日志不包含详细时延数据")
            return

        steps = range(len(self.timings))
        # 兼容新旧格式：使用辅助函数安全获取时延值
        trans_lats = [self._get_latency_ms(t, 'transport_latency') for t in self.timings]
        infer_lats = [self._get_latency_ms(t, 'inference_latency') for t in self.timings]
        total_lats = [self._get_latency_ms(t, 'total_latency') for t in self.timings]

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(steps, total_lats, color='gray', alpha=0.3, label='Total Loop')
        ax.plot(steps, trans_lats, color='orange', label='Transport (Network)')
        ax.plot(steps, infer_lats, color='blue', label='Inference (GPU)')
        
        ax.set_title("时延组成分析 (ms)")
        ax.set_xlabel("Step")
        ax.set_ylabel("Latency (ms)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 添加阈值线
        ax.axhline(100, color='r', linestyle='--', alpha=0.5)
        ax.text(0, 105, '100ms Alert', color='r', fontsize=8)
        
        st.pyplot(fig)
        plt.close(fig)
        
        # 显示统计信息
        if total_lats:
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("平均总延迟", f"{np.mean(total_lats):.1f} ms")
            col2.metric("平均传输延迟", f"{np.mean(trans_lats):.1f} ms")
            col3.metric("平均推理延迟", f"{np.mean(infer_lats):.1f} ms")
            col4.metric("最大总延迟", f"{np.max(total_lats):.1f} ms")

# --- Main ---
st.set_page_config(layout="wide", page_title="Inference Debugger")
st.title("🔬 推理深度诊断工具")

log_dir = Path(__file__).parent.parent / "realworld_deploy" / "server" / "log"
log_files = sorted(list(log_dir.glob("inference_log_*.json")), key=lambda x: x.stat().st_mtime, reverse=True)

if log_files:
    selected_file = st.sidebar.selectbox("选择日志", log_files, format_func=lambda x: x.name)
    if 'gui' not in st.session_state or st.session_state.get('last_log') != selected_file:
        st.session_state.gui = InferenceGUI(str(selected_file))
        st.session_state.last_log = selected_file
else:
    st.error("未找到日志文件")

if 'gui' in st.session_state and st.session_state.gui.valid:
    gui = st.session_state.gui
    
    tab1, tab2 = st.tabs(["📺 逐帧回放 (Visual & Action)", "📈 性能分析 (Latency)"])
    
    with tab1:
        idx = st.slider("Step", 0, len(gui.steps)-1, 0)
        gui.plot_replay_frame(idx)
        
    with tab2:
        gui.plot_latency_analysis()