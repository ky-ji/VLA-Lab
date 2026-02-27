"""
VLA-Lab Latency Analysis Viewer

Deep dive into timing metrics for VLA deployment.
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from typing import List, Dict, Any

import vlalab


def load_timing_data(run_path: Path) -> List[Dict[str, Any]]:
    """Load timing data from a run."""
    timing_data = []
    steps_path = run_path / "steps.jsonl"
    
    if steps_path.exists():
        with open(steps_path, "r") as f:
            for line in f:
                if line.strip():
                    step = json.loads(line)
                    timing_data.append(step.get("timing", {}))
    
    return timing_data


def get_latency_ms(timing_dict: Dict, key_base: str) -> float:
    """Get latency value in ms."""
    new_key = f"{key_base}_ms"
    if new_key in timing_dict and timing_dict[new_key] is not None:
        return timing_dict[new_key]
    if key_base in timing_dict and timing_dict[key_base] is not None:
        return timing_dict[key_base] * 1000
    return np.nan


def render():
    """Render the latency analysis page."""
    st.title("📈 时延深度分析")
    
    # Sidebar: show current runs directory
    runs_dir = vlalab.get_runs_dir()
    st.sidebar.markdown("### 日志目录")
    st.sidebar.code(str(runs_dir))
    
    # List projects
    projects = vlalab.list_projects()
    
    if not projects:
        st.info(f"未找到任何项目。日志目录: `{runs_dir}`")
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
    
    # Multi-select for comparison
    selected_runs = st.sidebar.multiselect(
        "选择运行 (可多选比较)",
        run_paths,
        default=[run_paths[0]] if run_paths else [],
        format_func=lambda p: f"{p.name} ({p.parent.name})"
    )
    
    if not selected_runs:
        st.info("请选择至少一个运行")
        return
    
    # Load data
    all_timing_data = {}
    for run_path in selected_runs:
        timing_data = load_timing_data(run_path)
        if timing_data:
            all_timing_data[run_path.name] = timing_data
    
    if not all_timing_data:
        st.warning("选中的运行没有时延数据")
        return
    
    # Analysis tabs
    tab1, tab2, tab3 = st.tabs(["📊 时序图", "📈 统计分布", "🔍 详细对比"])
    
    with tab1:
        st.markdown("### 时延时序图")
        
        fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(all_timing_data)))
        
        for (run_name, timing_list), color in zip(all_timing_data.items(), colors):
            steps = range(len(timing_list))
            
            trans_lats = [get_latency_ms(t, "transport_latency") for t in timing_list]
            infer_lats = [get_latency_ms(t, "inference_latency") for t in timing_list]
            total_lats = [get_latency_ms(t, "total_latency") for t in timing_list]
            
            axes[0].plot(steps, trans_lats, color=color, alpha=0.7, label=run_name)
            axes[1].plot(steps, infer_lats, color=color, alpha=0.7, label=run_name)
            axes[2].plot(steps, total_lats, color=color, alpha=0.7, label=run_name)
        
        axes[0].set_ylabel("Transport (ms)")
        axes[0].set_title("传输延迟 (网络)")
        axes[0].legend(loc='upper right')
        axes[0].grid(True, alpha=0.3)
        axes[0].axhline(50, color='orange', linestyle='--', alpha=0.5)
        
        axes[1].set_ylabel("Inference (ms)")
        axes[1].set_title("推理延迟 (GPU)")
        axes[1].legend(loc='upper right')
        axes[1].grid(True, alpha=0.3)
        
        axes[2].set_ylabel("Total (ms)")
        axes[2].set_xlabel("Step")
        axes[2].set_title("总回路延迟")
        axes[2].legend(loc='upper right')
        axes[2].grid(True, alpha=0.3)
        axes[2].axhline(100, color='red', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    
    with tab2:
        st.markdown("### 延迟分布")
        
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        
        for run_name, timing_list in all_timing_data.items():
            trans_lats = [get_latency_ms(t, "transport_latency") for t in timing_list]
            infer_lats = [get_latency_ms(t, "inference_latency") for t in timing_list]
            total_lats = [get_latency_ms(t, "total_latency") for t in timing_list]
            
            trans_valid = [x for x in trans_lats if not np.isnan(x)]
            infer_valid = [x for x in infer_lats if not np.isnan(x)]
            total_valid = [x for x in total_lats if not np.isnan(x)]
            
            if trans_valid:
                axes[0].hist(trans_valid, bins=30, alpha=0.5, label=run_name)
            if infer_valid:
                axes[1].hist(infer_valid, bins=30, alpha=0.5, label=run_name)
            if total_valid:
                axes[2].hist(total_valid, bins=30, alpha=0.5, label=run_name)
        
        axes[0].set_xlabel("Transport Latency (ms)")
        axes[0].set_ylabel("Count")
        axes[0].set_title("传输延迟分布")
        axes[0].legend()
        
        axes[1].set_xlabel("Inference Latency (ms)")
        axes[1].set_ylabel("Count")
        axes[1].set_title("推理延迟分布")
        axes[1].legend()
        
        axes[2].set_xlabel("Total Latency (ms)")
        axes[2].set_ylabel("Count")
        axes[2].set_title("总延迟分布")
        axes[2].legend()
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    
    with tab3:
        st.markdown("### 详细统计对比")
        
        # Create comparison table
        stats_data = []
        
        for run_name, timing_list in all_timing_data.items():
            trans_lats = [get_latency_ms(t, "transport_latency") for t in timing_list]
            infer_lats = [get_latency_ms(t, "inference_latency") for t in timing_list]
            total_lats = [get_latency_ms(t, "total_latency") for t in timing_list]
            
            trans_valid = [x for x in trans_lats if not np.isnan(x)]
            infer_valid = [x for x in infer_lats if not np.isnan(x)]
            total_valid = [x for x in total_lats if not np.isnan(x)]
            
            stats = {
                "运行": run_name,
                "步数": len(timing_list),
            }
            
            if trans_valid:
                stats["传输-平均(ms)"] = f"{np.mean(trans_valid):.1f}"
                stats["传输-P95(ms)"] = f"{np.percentile(trans_valid, 95):.1f}"
            
            if infer_valid:
                stats["推理-平均(ms)"] = f"{np.mean(infer_valid):.1f}"
                stats["推理-P95(ms)"] = f"{np.percentile(infer_valid, 95):.1f}"
            
            if total_valid:
                stats["总计-平均(ms)"] = f"{np.mean(total_valid):.1f}"
                stats["总计-P95(ms)"] = f"{np.percentile(total_valid, 95):.1f}"
                stats["总计-最大(ms)"] = f"{np.max(total_valid):.1f}"
            
            stats_data.append(stats)
        
        import pandas as pd
        df = pd.DataFrame(stats_data)
        st.dataframe(df, width='stretch')
        
        # Performance assessment
        st.markdown("### 性能评估")
        
        for run_name, timing_list in all_timing_data.items():
            total_lats = [get_latency_ms(t, "total_latency") for t in timing_list]
            total_valid = [x for x in total_lats if not np.isnan(x)]
            
            if total_valid:
                avg = np.mean(total_valid)
                p95 = np.percentile(total_valid, 95)
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown(f"**{run_name}**")
                
                with col2:
                    if avg < 50:
                        st.success(f"平均延迟: {avg:.1f}ms ✓")
                    elif avg < 100:
                        st.warning(f"平均延迟: {avg:.1f}ms")
                    else:
                        st.error(f"平均延迟: {avg:.1f}ms ✗")
                
                with col3:
                    if p95 < 100:
                        st.success(f"P95延迟: {p95:.1f}ms ✓")
                    elif p95 < 200:
                        st.warning(f"P95延迟: {p95:.1f}ms")
                    else:
                        st.error(f"P95延迟: {p95:.1f}ms ✗")
