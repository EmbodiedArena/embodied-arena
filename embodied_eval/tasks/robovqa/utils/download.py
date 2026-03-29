#!/usr/bin/env python3
"""
RoboVQA 数据集下载和准备工具
标准化接口，用于下载视频文件和准备数据集
"""
import os
import sys
import subprocess
from pathlib import Path
from typing import Optional, Set
from tqdm import tqdm
from loguru import logger

def check_missing_videos(
    dataset_path: str,
    video_dir: str,
    backup_video_dir: Optional[str] = None
) -> Set[str]:
    """
    检查缺失的视频文件
    
    Args:
        dataset_path: 数据集路径（HuggingFace Dataset格式）
        video_dir: 主要视频目录
        backup_video_dir: 备用视频目录（可选）
    
    Returns:
        缺失的视频文件名集合
    """
    try:
        from datasets import load_from_disk
        
        logger.info(f"加载数据集: {dataset_path}")
        dataset = load_from_disk(dataset_path)
        val_dataset = dataset
        
        # 获取所有需要的视频文件
        all_val_videos = set([val_dataset[i]['video'] for i in range(len(val_dataset))])
        
        # 检查本地文件（合并两个目录）
        local_files = set()
        if os.path.exists(video_dir):
            local_files.update(os.listdir(video_dir))
        if backup_video_dir and os.path.exists(backup_video_dir):
            local_files.update(os.listdir(backup_video_dir))
        
        missing = all_val_videos - local_files
        
        logger.info(f"📊 视频文件检查:")
        logger.info(f"   Val数据集需要的视频: {len(all_val_videos)}")
        logger.info(f"   本地存在的视频: {len(all_val_videos & local_files)}")
        logger.info(f"   缺失的视频: {len(missing)}")
        
        if missing:
            logger.info(f"\n缺失的视频文件示例（前10个）:")
            for i, video in enumerate(list(missing)[:10]):
                logger.info(f"   {i+1}. {video}")
        
        return missing
        
    except Exception as e:
        logger.error(f"检查失败: {e}")
        import traceback
        traceback.print_exc()
        return set()


def download_with_gsutil(
    bucket_path: str,
    local_path: str
) -> bool:
    """
    使用 gsutil 下载视频文件
    
    Args:
        bucket_path: GCS bucket路径
        local_path: 本地保存路径
    
    Returns:
        是否成功
    """
    logger.info(f"使用 gsutil 从 {bucket_path} 下载到 {local_path}")
    
    # 确保目标目录存在
    os.makedirs(local_path, exist_ok=True)
    
    # 使用 gsutil -m 进行并行下载
    cmd = ["gsutil", "-m", "cp", "-r", bucket_path, local_path]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.info("✅ 下载完成！")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ gsutil 执行失败: {e}")
        logger.error(f"stdout: {e.stdout}")
        logger.error(f"stderr: {e.stderr}")
        return False
    except FileNotFoundError:
        logger.error("❌ gsutil 未找到")
        return False


def download_with_python_gcs(
    bucket_name: str,
    prefix: str,
    local_path: str,
    missing_videos: Optional[Set[str]] = None
) -> bool:
    """
    使用 Python google-cloud-storage 库下载缺失的视频文件
    
    Args:
        bucket_name: GCS bucket名称
        prefix: 文件前缀路径
        local_path: 本地保存路径
        missing_videos: 缺失的视频文件集合
    
    Returns:
        是否成功
    """
    try:
        from google.cloud import storage
    except ImportError:
        logger.error("❌ google-cloud-storage 未安装")
        logger.error("安装方法: pip install google-cloud-storage")
        return False
    
    if missing_videos is None or len(missing_videos) == 0:
        logger.info("✅ 没有缺失的视频文件，无需下载")
        return True
    
    logger.info(f"使用 Python GCS 客户端从 {bucket_name}/{prefix} 下载 {len(missing_videos)} 个缺失的视频文件")
    logger.info(f"下载到: {local_path}")
    
    # 确保目标目录存在
    os.makedirs(local_path, exist_ok=True)
    
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        
        downloaded = 0
        failed = 0
        
        # 只下载缺失的视频文件
        for video_name in tqdm(missing_videos, desc="下载进度"):
            blob_path = f"{prefix}{video_name}"
            local_file_path = os.path.join(local_path, video_name)
            
            # 如果文件已存在，跳过
            if os.path.exists(local_file_path):
                continue
            
            try:
                blob = bucket.blob(blob_path)
                if blob.exists():
                    blob.download_to_filename(local_file_path)
                    downloaded += 1
                else:
                    logger.warning(f"⚠️  文件不存在于GCS: {video_name}")
                    failed += 1
            except Exception as e:
                logger.error(f"❌ 下载失败 {video_name}: {e}")
                failed += 1
        
        logger.info(f"\n✅ 下载完成！")
        logger.info(f"   成功下载: {downloaded} 个文件")
        if failed > 0:
            logger.info(f"   失败: {failed} 个文件")
        return True
        
    except Exception as e:
        logger.error(f"❌ 下载失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def download_robovqa_videos(
    dataset_path: str,
    video_dir: str,
    backup_video_dir: Optional[str] = None,
    bucket_name: str = "gdm-robovqa",
    bucket_prefix: str = "videos/",
    use_gsutil: bool = True
) -> bool:
    """
    下载 RoboVQA 视频文件（标准化接口）
    
    Args:
        dataset_path: 数据集路径
        video_dir: 视频保存目录
        backup_video_dir: 备用视频目录（可选）
        bucket_name: GCS bucket名称
        bucket_prefix: GCS文件前缀
        use_gsutil: 是否优先使用gsutil
    
    Returns:
        是否成功
    """
    logger.info("=" * 60)
    logger.info("RoboVQA 视频文件下载工具")
    logger.info("=" * 60)
    
    # 首先检查缺失的文件
    missing = check_missing_videos(dataset_path, video_dir, backup_video_dir)
    
    if not missing:
        logger.info("\n✅ 所有视频文件都已存在，无需下载！")
        return True
    
    logger.info(f"\n需要下载 {len(missing)} 个视频文件")
    
    # 方法1: 尝试使用 gsutil
    if use_gsutil:
        logger.info("\n方法1: 尝试使用 gsutil...")
        bucket_path = f"gs://{bucket_name}/{bucket_prefix}"
        if download_with_gsutil(bucket_path, video_dir):
            return True
    
    # 方法2: 尝试使用 Python GCS 客户端
    logger.info("\n方法2: 尝试使用 Python GCS 客户端...")
    if download_with_python_gcs(bucket_name, bucket_prefix, video_dir, missing):
        return True
    
    # 如果都失败了，提供安装指南
    logger.info("\n" + "=" * 60)
    logger.info("安装指南")
    logger.info("=" * 60)
    logger.info("\n方法1: 安装 gsutil (推荐)")
    logger.info("  conda install -c conda-forge google-cloud-sdk")
    logger.info("  或者")
    logger.info("  pip install gsutil")
    logger.info("\n方法2: 安装 Python GCS 客户端")
    logger.info("  pip install google-cloud-storage")
    logger.info("\n方法3: 手动安装 Google Cloud SDK")
    logger.info("  访问: https://cloud.google.com/sdk/docs/install")
    
    return False


def prepare_dataset(
    dataset_path: str,
    video_dir: str,
    backup_video_dir: Optional[str] = None,
    auto_download: bool = True
) -> bool:
    """
    准备 RoboVQA 数据集（标准化接口）
    包括检查数据集、检查视频文件、自动下载缺失的视频
    
    Args:
        dataset_path: 数据集路径
        video_dir: 视频目录
        backup_video_dir: 备用视频目录
        auto_download: 是否自动下载缺失的视频
    
    Returns:
        是否准备成功
    """
    logger.info("=" * 60)
    logger.info("准备 RoboVQA 数据集")
    logger.info("=" * 60)
    
    # 检查数据集路径
    if not os.path.exists(dataset_path):
        logger.error(f"❌ 数据集路径不存在: {dataset_path}")
        return False
    logger.info(f"✅ 数据集路径存在: {dataset_path}")
    
    # 检查视频目录
    if not os.path.exists(video_dir):
        logger.warning(f"⚠️  视频目录不存在: {video_dir}")
        if auto_download:
            logger.info("将尝试自动下载视频文件...")
        else:
            logger.error("请手动创建视频目录或设置 auto_download=True")
            return False
    else:
        logger.info(f"✅ 视频目录存在: {video_dir}")
    
    # 检查缺失的视频
    missing = check_missing_videos(dataset_path, video_dir, backup_video_dir)
    
    if missing and auto_download:
        logger.info(f"\n发现 {len(missing)} 个缺失的视频文件，开始自动下载...")
        success = download_robovqa_videos(
            dataset_path=dataset_path,
            video_dir=video_dir,
            backup_video_dir=backup_video_dir
        )
        if not success:
            logger.error("❌ 自动下载失败，请手动下载视频文件")
            return False
    
    if missing and not auto_download:
        logger.warning(f"⚠️  发现 {len(missing)} 个缺失的视频文件")
        logger.warning("请运行下载脚本或设置 auto_download=True")
        return False
    
    logger.info("\n✅ 数据集准备完成！")
    return True


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="RoboVQA 数据集下载和准备工具")
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="/home/n84416302/dataset/embodied-next_data/robovqa/RoboVQA_TF2HF",
        help="数据集路径"
    )
    parser.add_argument(
        "--video_dir",
        type=str,
        default="/home/n84416302/dataset/embodied-next_data/robovqa/videos",
        help="视频保存目录"
    )
    parser.add_argument(
        "--backup_video_dir",
        type=str,
        default=None,
        help="备用视频目录"
    )
    parser.add_argument(
        "--auto_download",
        action="store_true",
        help="自动下载缺失的视频文件"
    )
    parser.add_argument(
        "--check_only",
        action="store_true",
        help="仅检查，不下载"
    )
    
    args = parser.parse_args()
    
    if args.check_only:
        missing = check_missing_videos(args.dataset_path, args.video_dir, args.backup_video_dir)
        sys.exit(0 if not missing else 1)
    else:
        success = prepare_dataset(
            dataset_path=args.dataset_path,
            video_dir=args.video_dir,
            backup_video_dir=args.backup_video_dir,
            auto_download=args.auto_download
        )
        sys.exit(0 if success else 1)

