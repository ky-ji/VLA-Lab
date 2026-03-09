#!/usr/bin/env python3
"""
VLA-Lab Open-Loop Evaluation for RealWorld-VITA

使用 VLA-Lab 的统一评估框架对 VITA 模型进行开环评估。
通过 VitaAdapter 将 VITA Policy 适配到 VLA-Lab 接口，
并使用 AVAlohaDatasetLoader 加载 AVAloha 格式数据集。

用法:
    python examples/eval_vita_openloop.py \
        --vita_root /path/to/RealWorld-VITA \
        --config_path /path/to/train_config.yaml \
        --checkpoint_dir /path/to/step_0000020000 \
        --traj_ids 0 1 2 \
        --max_steps 300 \
        --save_plots_dir outputs/vita_eval/ \
        --device cuda:0

    # 使用默认路径 (stack_bowls_double_view):
    python examples/eval_vita_openloop.py
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

# =============================================================================
# 默认配置 - 根据你的环境修改
# =============================================================================

VITA_ROOT = "/home/jikangye/workspace/baselines/dp-baselines/RealWorld-VITA"

DEFAULT_CONFIG_PATH = (
    f"{VITA_ROOT}/ckpts/stack_bowls_double_view/"
    "vita-20260226_140930/logs/train_config_202602261409.yaml"
)
DEFAULT_CHECKPOINT_DIR = (
    f"{VITA_ROOT}/ckpts/stack_bowls_double_view/"
    "vita-20260226_140930/checkpoints/step_0000020000"
)

DEFAULT_TRAJ_IDS = [0, 1, 2]
DEFAULT_MAX_STEPS = 300
DEFAULT_SAVE_PLOTS_DIR = "outputs/vita_eval/"


def parse_args():
    parser = argparse.ArgumentParser(
        description="VLA-Lab Open-Loop Evaluation for RealWorld-VITA"
    )
    parser.add_argument(
        "--vita_root", type=str, default=VITA_ROOT,
        help="RealWorld-VITA 项目根目录"
    )
    parser.add_argument(
        "--config_path", type=str, default=DEFAULT_CONFIG_PATH,
        help="训练配置 YAML 路径"
    )
    parser.add_argument(
        "--checkpoint_dir", type=str, default=DEFAULT_CHECKPOINT_DIR,
        help="Checkpoint 目录路径 (包含 model.safetensors)"
    )
    parser.add_argument(
        "--traj_ids", type=int, nargs="+", default=DEFAULT_TRAJ_IDS,
        help="要评估的轨迹 ID 列表"
    )
    parser.add_argument(
        "--action_horizon", type=int, default=None,
        help="动作预测步长 (默认从配置读取)"
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


def setup_sys_path(vita_root: str):
    """Ensure RealWorld-VITA packages are importable."""
    vita_root = Path(vita_root)
    paths_to_add = [
        str(vita_root),
        str(vita_root / "lerobot"),
        str(vita_root / "gym-av-aloha"),
    ]
    for p in paths_to_add:
        if p not in sys.path:
            sys.path.insert(0, p)


def main():
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    logging.info("=" * 60)
    logging.info("VLA-Lab Open-Loop Evaluation for RealWorld-VITA")
    logging.info("=" * 60)

    # Step 0: Setup sys.path
    setup_sys_path(args.vita_root)

    # Step 1: Load VITA policy
    logging.info(f"加载 VITA 配置: {args.config_path}")
    logging.info(f"加载 Checkpoint: {args.checkpoint_dir}")

    from vlalab.eval.adapters.vita_adapter import load_vita_policy, VitaAdapter
    policy, cfg = load_vita_policy(
        config_path=args.config_path,
        checkpoint_dir=args.checkpoint_dir,
        device=args.device,
    )

    # Step 2: Wrap with VitaAdapter
    adapter = VitaAdapter(policy, cfg, device=args.device)
    modality = adapter.get_modality_config()

    logging.info(f"VLA-Lab adapter modality config:")
    logging.info(f"  State keys:    {modality.state_keys}")
    logging.info(f"  Action keys:   {modality.action_keys}")
    logging.info(f"  Image keys:    {modality.image_keys}")
    logging.info(f"  Action horizon: {modality.action_horizon}")
    logging.info(f"  Action dim:    {modality.action_dim}")

    # Step 3: Build AVAloha dataset loader
    from vlalab.eval.avaloha_loader import AVAlohaDatasetLoader
    dataset_root = cfg.task.dataset_root
    dataset_loader = AVAlohaDatasetLoader(dataset_root=dataset_root)
    logging.info(f"数据集路径: {dataset_root}")
    logging.info(f"数据集轨迹数: {len(dataset_loader)}")

    # Step 4: Build evaluator and run
    from vlalab.eval.open_loop_eval import OpenLoopEvaluator, EvalConfig

    evaluator = OpenLoopEvaluator(
        policy=adapter,
        dataset_path=str(dataset_root),
        dataset_format="custom",
        dataset_loader=dataset_loader,
    )

    action_horizon = args.action_horizon or cfg.policy.action_horizon

    logging.info(f"\n开始评估 {len(args.traj_ids)} 条轨迹...")
    logging.info(f"  Trajectory IDs: {args.traj_ids}")
    logging.info(f"  Action Horizon: {action_horizon}")
    logging.info(f"  Max Steps:      {args.max_steps}")

    results = evaluator.evaluate(
        traj_ids=args.traj_ids,
        max_steps=args.max_steps,
        action_horizon=action_horizon,
        save_plots_dir=args.save_plots_dir,
    )

    # Print results
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

    # Save JSON results
    if args.save_plots_dir:
        results_path = Path(args.save_plots_dir) / "results.json"
        evaluator.evaluate_and_save(
            output_path=str(results_path),
            traj_ids=args.traj_ids,
            max_steps=args.max_steps,
            action_horizon=action_horizon,
        )
        logging.info(f"\n图表保存到: {args.save_plots_dir}")
        logging.info(f"结果 JSON 保存到: {results_path}")

    logging.info("\n✅ 评估完成!")
    return results


if __name__ == "__main__":
    main()
