#!/usr/bin/env python3
"""
转换旧版日志为 VLA-Lab 格式

示例：转换 RealWorld-DP 和 Isaac-GR00T 的旧版推理日志。
"""

from pathlib import Path
from vlalab.adapters.converter import convert_legacy_log, detect_log_format


def convert_dp_logs():
    """转换 RealWorld-DP 日志"""
    log_dir = Path("/home/jikangye/workspace/baselines/vla-baselines/RealWorld-DP/realworld_deploy/server/log")
    
    if not log_dir.exists():
        print(f"DP 日志目录不存在: {log_dir}")
        return
    
    # 查找所有日志文件
    log_files = list(log_dir.glob("inference_log_*.json"))
    print(f"找到 {len(log_files)} 个 DP 日志文件")
    
    for log_file in log_files[:3]:  # 只转换前 3 个作为示例
        output_dir = log_dir / "converted" / f"run_{log_file.stem}"
        
        print(f"\n转换: {log_file.name}")
        print(f"  -> {output_dir}")
        
        try:
            stats = convert_legacy_log(log_file, output_dir, "dp")
            print(f"  ✓ 完成: {stats['steps']} 步, {stats['images']} 张图像")
        except Exception as e:
            print(f"  ✗ 失败: {e}")


def convert_groot_logs():
    """转换 Isaac-GR00T 日志"""
    log_dir = Path("/home/jikangye/workspace/baselines/vla-baselines/Isaac-GR00T/realworld_deploy/server/log")
    
    if not log_dir.exists():
        print(f"GR00T 日志目录不存在: {log_dir}")
        return
    
    # 查找所有日志文件
    log_files = list(log_dir.glob("inference_log_groot_*.json"))
    print(f"找到 {len(log_files)} 个 GR00T 日志文件")
    
    for log_file in log_files[:3]:  # 只转换前 3 个作为示例
        output_dir = log_dir / "converted" / f"run_{log_file.stem}"
        
        print(f"\n转换: {log_file.name}")
        print(f"  -> {output_dir}")
        
        try:
            stats = convert_legacy_log(log_file, output_dir, "groot")
            print(f"  ✓ 完成: {stats['steps']} 步, {stats['images']} 张图像")
        except Exception as e:
            print(f"  ✗ 失败: {e}")


def auto_detect_and_convert(log_path: str):
    """自动检测格式并转换"""
    log_path = Path(log_path)
    
    if not log_path.exists():
        print(f"文件不存在: {log_path}")
        return
    
    # 自动检测格式
    format_type = detect_log_format(log_path)
    print(f"检测到格式: {format_type}")
    
    # 转换
    output_dir = log_path.parent / f"run_{log_path.stem}"
    stats = convert_legacy_log(log_path, output_dir, format_type)
    
    print(f"转换完成:")
    print(f"  - 步数: {stats['steps']}")
    print(f"  - 图像: {stats['images']}")
    print(f"  - 输出: {stats['output_dir']}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # 转换指定文件
        auto_detect_and_convert(sys.argv[1])
    else:
        # 转换示例
        print("=" * 60)
        print("转换 RealWorld-DP 日志")
        print("=" * 60)
        convert_dp_logs()
        
        print("\n" + "=" * 60)
        print("转换 Isaac-GR00T 日志")
        print("=" * 60)
        convert_groot_logs()
