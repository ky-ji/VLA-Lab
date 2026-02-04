#!/usr/bin/env python3
"""
VLA-Lab 基本使用示例

展示如何在推理循环中使用 RunLogger。
"""

import numpy as np
import time
from vlalab import RunLogger


def simulate_inference():
    """模拟推理过程"""
    # 模拟观测
    state = np.random.randn(8).astype(np.float32)  # [x, y, z, qx, qy, qz, qw, gripper]
    image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # 模拟推理延迟
    time.sleep(0.05)
    
    # 模拟动作输出
    action_chunk = np.random.randn(8, 8).astype(np.float32)
    
    return state, image, action_chunk


def main():
    # 初始化 logger
    logger = RunLogger(
        run_dir="example_runs/basic_example",
        model_name="example_model",
        model_path="/path/to/checkpoint",
        model_type="diffusion_policy",
        task_name="pick_and_place",
        robot_name="franka",
        inference_freq=10.0,
    )
    
    print(f"开始记录到: {logger.run_dir}")
    
    # 模拟推理循环
    for step_idx in range(20):
        # 记录时间戳
        t_start = time.time()
        
        # 模拟推理
        state, image, action_chunk = simulate_inference()
        
        t_end = time.time()
        
        # 记录这一步
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
    
    print(f"\n记录完成！")
    print(f"  - 总步数: {logger.step_count}")
    print(f"  - 日志目录: {logger.run_dir}")
    print(f"\n查看日志:")
    print(f"  vlalab view")


if __name__ == "__main__":
    main()
