#!/usr/bin/env python3
"""
VLA-Lab Open-Loop Evaluation for Isaac GR00T

使用 VLA-Lab 的统一评估框架对 GR00T 模型进行开环评估。
通过 GR00TAdapter 将 GR00T Policy 适配到 VLA-Lab 接口，
并使用 LeRobotDatasetLoader 加载 GR00T 格式的 LeRobot v2 数据集。

用法:
    # 方式 1: 直接运行（使用下方默认配置）
    python examples/eval_groot_openloop.py

    # 方式 2: 通过命令行参数
    python examples/eval_groot_openloop.py \
        --model_path /path/to/checkpoint \
        --dataset_path /path/to/dataset \
        --embodiment_tag NEW_EMBODIMENT \
        --traj_ids 0 1 2 \
        --action_horizon 16 \
        --max_steps 300 \
        --save_plots_dir outputs/groot_eval/
        
        
    python ../VLA-Lab/examples/eval_groot_openloop.py \
    --model_path ckpts/GR00T-N1.6-3B_assembly_things/checkpoint-100000 \
    --dataset_path /data1/vla-data/processed/GR00T/000204_assembly_things \
    --embodiment_tag FRANKA \
    --modality_config_path examples/assembly_things/assembly_things_config.py \
    --traj_ids 10 310 670 \
    --action_horizon 8 \
    --max_steps 300 \
    --save_plots_dir outputs/vlalab_eval/ \
    --device cuda:0
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

# =============================================================================
# 默认配置 - 根据你的环境修改
# =============================================================================

# 数据集路径（LeRobot v2 格式）
DEFAULT_DATASET_PATH = "demo_data/cube_to_bowl_5"

# 模型 checkpoint 路径
DEFAULT_MODEL_PATH = None  # None 表示使用 PolicyClient 连接远程推理服务

# Embodiment tag
DEFAULT_EMBODIMENT_TAG = "NEW_EMBODIMENT"

# 评估参数
DEFAULT_TRAJ_IDS = [0]
DEFAULT_ACTION_HORIZON = 16
DEFAULT_MAX_STEPS = 200

# 输出路径
DEFAULT_SAVE_PLOTS_DIR = "outputs/groot_eval/"


def parse_args():
    parser = argparse.ArgumentParser(
        description="VLA-Lab Open-Loop Evaluation for Isaac GR00T"
    )
    parser.add_argument(
        "--model_path", type=str, default=DEFAULT_MODEL_PATH,
        help="模型 checkpoint 路径 (None=使用 PolicyClient)"
    )
    parser.add_argument(
        "--dataset_path", type=str, default=DEFAULT_DATASET_PATH,
        help="LeRobot v2 格式数据集路径"
    )
    parser.add_argument(
        "--embodiment_tag", type=str, default=DEFAULT_EMBODIMENT_TAG,
        help="Embodiment tag (如 NEW_EMBODIMENT, FRANKA, GR1 等)"
    )
    parser.add_argument(
        "--modality_config_path", type=str, default=None,
        help="自定义 modality 配置 Python 文件路径 (可选)"
    )
    parser.add_argument(
        "--host", type=str, default="127.0.0.1",
        help="PolicyClient 远程推理服务地址"
    )
    parser.add_argument(
        "--port", type=int, default=5555,
        help="PolicyClient 远程推理服务端口"
    )
    parser.add_argument(
        "--traj_ids", type=int, nargs="+", default=DEFAULT_TRAJ_IDS,
        help="要评估的轨迹 ID 列表"
    )
    parser.add_argument(
        "--action_horizon", type=int, default=DEFAULT_ACTION_HORIZON,
        help="动作预测步长"
    )
    parser.add_argument(
        "--max_steps", type=int, default=DEFAULT_MAX_STEPS,
        help="每条轨迹最大评估步数"
    )
    parser.add_argument(
        "--save_plots_dir", type=str, default=DEFAULT_SAVE_PLOTS_DIR,
        help="保存评估图表的目录"
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0",
        help="推理设备"
    )
    return parser.parse_args()


def _resolve_embodiment_tag(tag_str: str):
    """将字符串解析为 EmbodimentTag，支持枚举名(FRANKA)和枚举值(franka)"""
    from gr00t.data.embodiment_tags import EmbodimentTag
    # 先按枚举名查找 (如 FRANKA, NEW_EMBODIMENT)
    try:
        return EmbodimentTag[tag_str]
    except KeyError:
        pass
    # 再按枚举值查找 (如 franka, new_embodiment)
    try:
        return EmbodimentTag(tag_str)
    except ValueError:
        pass
    # 最后尝试大小写变换
    try:
        return EmbodimentTag[tag_str.upper()]
    except KeyError:
        pass
    raise ValueError(
        f"'{tag_str}' 不是有效的 EmbodimentTag。"
        f"可用值: {[e.name for e in EmbodimentTag]}"
    )


def load_groot_policy(args):
    """加载 GR00T Policy（本地模型或远程客户端）"""
    embodiment_tag = _resolve_embodiment_tag(args.embodiment_tag)

    if args.model_path is not None:
        import torch
        from gr00t.policy.gr00t_policy import Gr00tPolicy

        device = args.device if torch.cuda.is_available() else "cpu"
        logging.info(f"加载本地 GR00T Policy: {args.model_path}")
        logging.info(f"  Embodiment: {embodiment_tag}")
        logging.info(f"  Device: {device}")

        policy = Gr00tPolicy(
            embodiment_tag=embodiment_tag,
            model_path=args.model_path,
            device=device,
        )
    else:
        from gr00t.policy.server_client import PolicyClient

        logging.info(f"连接远程 GR00T Policy: {args.host}:{args.port}")
        policy = PolicyClient(host=args.host, port=args.port)

        if hasattr(policy, "ping") and not policy.ping():
            raise RuntimeError(
                f"无法连接到推理服务 {args.host}:{args.port}\n"
                "请先启动服务: uv run python gr00t/eval/run_gr00t_server.py ..."
            )

    return policy, embodiment_tag


def load_modality_config(args, embodiment_tag):
    """加载自定义 modality 配置（如果指定了配置文件）"""
    if args.modality_config_path:
        logging.info(f"加载自定义 modality 配置: {args.modality_config_path}")
        import importlib.util
        spec = importlib.util.spec_from_file_location("modality_config", args.modality_config_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        logging.info("自定义 modality 配置已注册")


def main():
    args = parse_args()

    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    logging.info("=" * 60)
    logging.info("VLA-Lab Open-Loop Evaluation for Isaac GR00T")
    logging.info("=" * 60)

    # Step 1: 加载 modality config（如有自定义）
    embodiment_tag = _resolve_embodiment_tag(args.embodiment_tag)
    load_modality_config(args, embodiment_tag)

    # Step 2: 加载 GR00T Policy
    policy, embodiment_tag = load_groot_policy(args)
    modality_config = policy.get_modality_config()
    logging.info(f"Policy modality config:\n{modality_config}")

    # Step 3: 使用 VLA-Lab 的 GR00TAdapter 包装
    from vlalab.eval.adapters.groot_adapter import GR00TAdapter
    adapter = GR00TAdapter(policy, embodiment_tag=args.embodiment_tag)
    vlalab_modality = adapter.get_modality_config()

    logging.info(f"VLA-Lab adapter modality config:")
    logging.info(f"  State keys:    {vlalab_modality.state_keys}")
    logging.info(f"  Action keys:   {vlalab_modality.action_keys}")
    logging.info(f"  Image keys:    {vlalab_modality.image_keys}")
    logging.info(f"  Language keys: {vlalab_modality.language_keys}")
    logging.info(f"  Action horizon: {vlalab_modality.action_horizon}")

    # Step 4: 构建 LeRobot 数据加载器
    from vlalab.eval.lerobot_loader import LeRobotDatasetLoader
    dataset_loader = LeRobotDatasetLoader(
        dataset_path=args.dataset_path,
        modality_configs=modality_config,
        embodiment_tag=embodiment_tag,
    )
    logging.info(f"数据集轨迹数: {len(dataset_loader)}")

    # Step 5: 构建评估器并运行
    from vlalab.eval.open_loop_eval import (
        OpenLoopEvaluator,
        EvalConfig,
        evaluate_trajectory,
        plot_trajectory_results,
    )

    evaluator = OpenLoopEvaluator(
        policy=adapter,
        dataset_path=args.dataset_path,
        dataset_format="lerobot",
        dataset_loader=dataset_loader,
    )

    # 获取任务描述
    task_desc = dataset_loader.get_task_description(args.traj_ids[0])
    logging.info(f"任务描述: {task_desc}")

    logging.info(f"\n开始评估 {len(args.traj_ids)} 条轨迹...")
    logging.info(f"  Trajectory IDs: {args.traj_ids}")
    logging.info(f"  Action Horizon: {args.action_horizon}")
    logging.info(f"  Max Steps:      {args.max_steps}")

    results = evaluator.evaluate(
        traj_ids=args.traj_ids,
        max_steps=args.max_steps,
        action_horizon=args.action_horizon,
        save_plots_dir=args.save_plots_dir,
        task_description=task_desc,
    )

    # 打印结果
    logging.info("\n" + "=" * 60)
    logging.info("评估结果")
    logging.info("=" * 60)
    logging.info(f"评估轨迹数: {results['num_trajectories']}")

    if "avg_mse" in results:
        logging.info(f"平均 MSE: {results['avg_mse']:.6f}")
        logging.info(f"平均 MAE: {results['avg_mae']:.6f}")

    for r in results["results"]:
        logging.info(
            f"  轨迹 {r['trajectory_id']}: "
            f"MSE={r['mse']:.6f}, MAE={r['mae']:.6f}, "
            f"步数={r['num_steps']}"
        )

    if args.save_plots_dir:
        logging.info(f"\n图表保存到: {args.save_plots_dir}")

    # 保存 JSON 结果
    results_path = Path(args.save_plots_dir) / "results.json"
    evaluator.evaluate_and_save(
        output_path=str(results_path),
        traj_ids=args.traj_ids,
        max_steps=args.max_steps,
        action_horizon=args.action_horizon,
        task_description=task_desc,
    )
    logging.info(f"结果 JSON 保存到: {results_path}")

    logging.info("\n✅ 评估完成!")
    return results


if __name__ == "__main__":
    main()
