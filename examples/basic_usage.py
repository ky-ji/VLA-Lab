#!/usr/bin/env python3
"""
VLA-Lab Basic Usage Examples

Demonstrates two API styles:
1. Simple API (recommended for most users)
2. Advanced API (RunLogger, for fine-grained control)
"""

import numpy as np
import time


def simulate_inference():
    """模拟推理过程"""
    # 模拟观测
    state = np.random.randn(8).astype(np.float32)  # [x, y, z, qx, qy, qz, qw, gripper]
    image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # 模拟推理延迟
    time.sleep(0.03)
    
    # 模拟动作输出
    action_chunk = np.random.randn(8, 8).astype(np.float32)
    
    return state, image, action_chunk


# ============================================================================
# Method 1: Simple API (Recommended)
# ============================================================================

def example_simple_api():
    """简洁 API 示例"""
    import vlalab
    
    # 初始化一个 run
    run = vlalab.init(
        project="pick_and_place",
        config={
            "model": "diffusion_policy",
            "action_horizon": 8,
            "inference_freq": 10,
            "robot": "franka",
        },
    )
    
    # Access config via run.config.xxx
    print(f"模型: {run.config.model}")
    print(f"Action Horizon: {run.config.action_horizon}")
    
    # 模拟推理循环
    for step in range(10):
        t_start = time.time()
        
        # 模拟推理
        state, image, action_chunk = simulate_inference()
        
        t_end = time.time()
        latency_ms = (t_end - t_start) * 1000
        
        # 记录这一步（简洁的字典格式）
        vlalab.log({
            "state": state.tolist(),
            "action": action_chunk.tolist(),
            "images": {"front": image},
            "inference_latency_ms": latency_ms,
        })
        
        print(f"Step {step}: latency = {latency_ms:.1f}ms")
    
    # 结束（或程序退出时自动调用）
    vlalab.finish()
    
    print(f"\n✅ 简洁 API 示例完成！")
    print(f"   日志保存到: {run.run_dir}")


# ============================================================================
# 方式 2: 高级 API（RunLogger）
# ============================================================================

def example_advanced_api():
    """高级 API 示例"""
    from vlalab import RunLogger
    
    # 初始化 logger
    logger = RunLogger(
        run_dir="example_runs/advanced_example",
        model_name="diffusion_policy",
        model_path="/path/to/checkpoint",
        model_type="diffusion_policy",
        task_name="pick_and_place",
        robot_name="franka",
        inference_freq=10.0,
    )
    
    print(f"开始记录到: {logger.run_dir}")
    
    # 模拟推理循环
    for step_idx in range(10):
        # 记录时间戳
        t_start = time.time()
        
        # 模拟推理
        state, image, action_chunk = simulate_inference()
        
        t_end = time.time()
        
        # 记录这一步（完整控制）
        logger.log_step(
            step_idx=step_idx,
            state=state.tolist(),
            pose=state[:7].tolist(),
            gripper=float(state[7]),
            action=action_chunk.tolist(),
            images={"front_camera": image},
            timing={
                "infer_start": t_start,
                "infer_end": t_end,
                "inference_latency_ms": (t_end - t_start) * 1000,
            },
        )
        
        print(f"Step {step_idx}: action shape = {action_chunk.shape}")
    
    # 关闭 logger
    logger.close()
    
    print(f"\n✅ 高级 API 示例完成！")
    print(f"   总步数: {logger.step_count}")
    print(f"   日志目录: {logger.run_dir}")


# ============================================================================
# 方式 3: 使用 with 语句（上下文管理器）
# ============================================================================

def example_context_manager():
    """使用 with 语句的示例"""
    import vlalab
    
    # 使用 with 语句，自动处理 finish()
    with vlalab.Run(
        project="context_example",
        config={"model": "groot", "task": "assembly"},
    ) as run:
        
        for step in range(5):
            state, image, action = simulate_inference()
            
            run.log({
                "state": state.tolist(),
                "action": action.tolist(),
                "images": {"cam": image},
            })
            
            print(f"Step {step}")
    
    # with 退出时自动 finish()
    print(f"\n✅ 上下文管理器示例完成！")


def main():
    print("=" * 60)
    print("VLA-Lab 使用示例")
    print("=" * 60)
    
    print("\n--- 示例 1: 简洁 API（推荐）---")
    example_simple_api()
    
    print("\n" + "=" * 60)
    print("\n--- 示例 2: 高级 API（RunLogger）---")
    example_advanced_api()
    
    print("\n" + "=" * 60)
    print("\n--- 示例 3: 上下文管理器 ---")
    example_context_manager()
    
    print("\n" + "=" * 60)
    print("\n查看日志:")
    print("  vlalab view")
    print("\n转换旧版日志:")
    print("  vlalab convert <input_path> -o <output_dir>")


if __name__ == "__main__":
    main()
