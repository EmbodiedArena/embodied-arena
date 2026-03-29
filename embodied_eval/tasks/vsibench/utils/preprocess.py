#!/usr/bin/env python3
"""
VSI-Bench 数据预处理脚本
解压缩视频文件并转换数据格式
"""
import os
import json
import zipfile
from pathlib import Path
from typing import List, Dict, Any, Optional
from tqdm import tqdm
from datasets import Dataset, DatasetDict
from loguru import logger


def extract_zip_files(
    raw_dir: str,
    video_dir: str
) -> bool:
    """
    解压缩视频 zip 文件
    
    Args:
        raw_dir: 原始数据目录（包含 zip 文件）
        video_dir: 视频保存目录
    
    Returns:
        是否成功
    """
    logger.info("=" * 60)
    logger.info("解压缩视频文件")
    logger.info("=" * 60)
    
    # 需要解压的 zip 文件
    zip_files = ["scannet.zip", "scannetpp.zip", "arkitscenes.zip"]
    
    # 确保视频目录存在
    os.makedirs(video_dir, exist_ok=True)
    
    for zip_name in zip_files:
        zip_path = os.path.join(raw_dir, zip_name)
        
        if not os.path.exists(zip_path):
            logger.warning(f"⚠️  zip 文件不存在: {zip_path}")
            continue
        
        logger.info(f"\n解压缩: {zip_name}")
        
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # 获取所有文件列表
                file_list = zip_ref.namelist()
                
                # 解压所有文件
                for file_info in tqdm(file_list, desc=f"解压 {zip_name}"):
                    zip_ref.extract(file_info, video_dir)
            
            logger.info(f"✅ {zip_name} 解压完成")
        except Exception as e:
            logger.error(f"❌ 解压失败 {zip_name}: {e}")
            return False
    
    logger.info("\n✅ 所有视频文件解压完成！")
    return True


def load_jsonl(jsonl_path: str) -> List[Dict[str, Any]]:
    """
    加载 JSONL 文件
    
    Args:
        jsonl_path: JSONL 文件路径
    
    Returns:
        数据列表
    """
    data = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line.strip()))
    return data


def convert_to_hf_dataset(
    jsonl_path: str,
    output_path: str
) -> DatasetDict:
    """
    将 JSONL 数据转换为 HuggingFace Dataset 格式
    
    Args:
        jsonl_path: JSONL 文件路径
        output_path: 输出路径（HuggingFace Dataset 格式）
    
    Returns:
        处理后的数据集
    """
    logger.info("=" * 60)
    logger.info("转换数据格式")
    logger.info("=" * 60)
    
    # 加载 JSONL 数据
    logger.info(f"加载数据: {jsonl_path}")
    data = load_jsonl(jsonl_path)
    logger.info(f"总共 {len(data)} 条数据")
    
    # 转换为标准格式
    processed_data = []
    for item in tqdm(data, desc="处理数据"):
        processed_item = {
            "id": item.get("id", 0),
            "question": item.get("question", ""),
            "ground_truth": item.get("ground_truth", ""),
            "question_type": item.get("question_type", ""),
            "dataset": item.get("dataset", ""),
            "scene_name": item.get("scene_name", ""),
        }
        
        # 如果有选项（多选题）
        if "options" in item and item["options"] is not None:
            processed_item["options"] = item["options"]
        else:
            processed_item["options"] = None
        
        processed_data.append(processed_item)
    
    # 创建数据集
    dataset = Dataset.from_list(processed_data)
    dataset_dict = DatasetDict({"test": dataset})
    
    # 保存数据集
    logger.info(f"\n保存数据集到: {output_path}")
    os.makedirs(output_path, exist_ok=True)
    dataset_dict.save_to_disk(output_path)
    
    logger.info("✅ 数据转换完成！")
    logger.info(f"   数据集路径: {output_path}")
    logger.info(f"   样本数量: {len(dataset)}")
    
    return dataset_dict


def preprocess_vsi_bench(
    raw_dir: str,
    output_dir: str,
    video_dir: Optional[str] = None
) -> bool:
    """
    完整的预处理流程：解压缩 + 数据转换
    
    Args:
        raw_dir: 原始数据目录（包含 zip 文件和 test.jsonl）
        output_dir: 输出目录（HuggingFace Dataset 格式）
        video_dir: 视频保存目录（如果为 None，则使用 output_dir）
    
    Returns:
        是否成功
    """
    logger.info("=" * 60)
    logger.info("VSI-Bench 数据预处理")
    logger.info("=" * 60)
    
    if video_dir is None:
        video_dir = output_dir
    
    # 步骤1: 解压缩视频文件
    logger.info("\n步骤1: 解压缩视频文件...")
    if not extract_zip_files(raw_dir, video_dir):
        logger.error("❌ 解压缩失败")
        return False
    
    # 步骤2: 转换数据格式
    logger.info("\n步骤2: 转换数据格式...")
    jsonl_path = os.path.join(raw_dir, "test.jsonl")
    if not os.path.exists(jsonl_path):
        logger.error(f"❌ JSONL 文件不存在: {jsonl_path}")
        return False
    
    try:
        convert_to_hf_dataset(jsonl_path, output_dir)
    except Exception as e:
        logger.error(f"❌ 数据转换失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    logger.info("\n" + "=" * 60)
    logger.info("✅ 预处理完成！")
    logger.info("=" * 60)
    logger.info(f"数据集路径: {output_dir}")
    logger.info(f"视频目录: {video_dir}")
    
    return True


if __name__ == "__main__":
    import argparse
    from typing import Optional
    
    parser = argparse.ArgumentParser(description="VSI-Bench 数据预处理脚本")
    parser.add_argument(
        "--raw_dir",
        type=str,
        default="/home/n84416302/dataset/embodied-next_data/vsi-bench-raw",
        help="原始数据目录（包含 zip 文件和 test.jsonl）"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/n84416302/dataset/embodied-next_data/vsi-bench",
        help="输出目录（HuggingFace Dataset 格式）"
    )
    parser.add_argument(
        "--video_dir",
        type=str,
        default=None,
        help="视频保存目录（如果为 None，则使用 output_dir）"
    )
    parser.add_argument(
        "--extract_only",
        action="store_true",
        help="仅解压缩，不转换数据格式"
    )
    parser.add_argument(
        "--convert_only",
        action="store_true",
        help="仅转换数据格式，不解压缩"
    )
    
    args = parser.parse_args()
    
    try:
        if args.extract_only:
            # 仅解压缩
            video_dir = args.video_dir or args.output_dir
            success = extract_zip_files(args.raw_dir, video_dir)
        elif args.convert_only:
            # 仅转换数据格式
            jsonl_path = os.path.join(args.raw_dir, "test.jsonl")
            success = convert_to_hf_dataset(jsonl_path, args.output_dir) is not None
        else:
            # 完整流程
            success = preprocess_vsi_bench(
                args.raw_dir,
                args.output_dir,
                args.video_dir
            )
        
        if success:
            logger.info("✅ 处理成功完成！")
        else:
            logger.error("❌ 处理失败")
            exit(1)
    except Exception as e:
        logger.error(f"❌ 处理失败: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
