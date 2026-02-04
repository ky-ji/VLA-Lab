#!/usr/bin/env python3
"""
VLA-Lab 快速入门示例

最简单的使用方式，类似 SwanLab。
"""

import vlalab
import numpy as np

# 初始化
run = vlalab.init(
    project="my-vla-project",
    config={
        "model": "diffusion_policy",
        "action_horizon": 8,
    },
)

print(f"Action Horizon: {run.config.action_horizon}")

# 模拟推理循环
for step in range(10):
    # 模拟数据
    state = np.random.randn(8).tolist()
    action = np.random.randn(8).tolist()
    image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # 记录
    vlalab.log({
        "state": state,
        "action": action,
        "images": {"front": image},
        "inference_latency_ms": 32.5,
    })
    
    print(f"Step {step} logged")

# 结束
vlalab.finish()
