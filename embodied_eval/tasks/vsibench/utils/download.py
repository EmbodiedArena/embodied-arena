#!/usr/bin/env python3
"""
VSI-Bench 数据集下载工具
从 HuggingFace 下载数据集
"""
import os
import subprocess
import sys
from pathlib import Path
from loguru import logger

# HuggingFace 数据集名称
HF_DATASET_NAME = "nyu-visionx/VSI-Bench"


def download_vsi_bench(
    local_dir: str
) -> bool:
    """
    从 HuggingFace 下载 VSI-Bench 数据集
    
    Args:
        local_dir: 本地保存目录
    
    Returns:
        是否成功
    """
    logger.info("=" * 60)
    logger.info("VSI-Bench 数据集下载")
    logger.info("=" * 60)
    logger.info(f"数据集: {HF_DATASET_NAME}")
    logger.info(f"保存到: {local_dir}")
    
    # 确保目录存在
    os.makedirs(local_dir, exist_ok=True)
    
    # 使用 hf download 命令下载
    cmd = [
        "hf", "download",
        HF_DATASET_NAME,
        "--repo-type", "dataset",
        "--local-dir", local_dir
    ]
    
    try:
        logger.info("\n开始下载...")
        result = subprocess.run(cmd, check=True, capture_output=False)
        logger.info("\n✅ 下载完成！")
        logger.info(f"数据集保存在: {local_dir}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ 下载失败: {e}")
        return False
    except FileNotFoundError:
        logger.error("❌ hf 命令未找到")
        logger.error("请安装: pip install huggingface-hub")
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="VSI-Bench 数据集下载工具")
    parser.add_argument(
        "--local_dir",
        type=str,
        default="/home/n84416302/dataset/embodied-next_data/vsi-bench-raw",
        help="本地保存目录"
    )
    
    args = parser.parse_args()
    
    success = download_vsi_bench(args.local_dir)
    sys.exit(0 if success else 1)
