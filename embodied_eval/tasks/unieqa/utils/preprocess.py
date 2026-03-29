#!/usr/bin/env python3
"""
UniEQA Dataset Preprocessing Utilities

This script provides utilities for preprocessing the complete UniEQA dataset.
"""

import os
import json
import re
import argparse
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd
from PIL import Image
from tqdm import tqdm

MAX_IMAGES_PER_SAMPLE = 15

def load_unieqa_data(data_root: str) -> List[Dict[str, Any]]:
    """
    Load UniEQA data from Part1 ~ Part6 data.json files

    Args:
        data_root: Root directory of UniEQA dataset

    Returns:
        List of processed samples
    """
    all_samples = []
    data_root_path = Path(data_root)

    # 1. 递归找到所有的 data.json 文件
    json_files = sorted(list(data_root_path.glob("**/core/data.json")))
    print(f"🚀 发现 {len(json_files)} 个配置文件，开始处理数据...")

    total_data_count = 0
    for json_path in json_files:
        # 获取 Part 信息
        part_name = "unknown"
        for part in ["Part1", "Part2", "Part3", "Part4", "Part5", "Part6"]:
            if part in str(json_path):
                part_name = part
                break
        
        # 获取能力维度信息 (取 core 的上一级目录名)
        capability_dimension = json_path.parent.parent.name
        
        # 图片目录通常是相对于 data.json 的 images 目录
        images_dir = json_path.parent / "images"

        with open(json_path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
            except Exception as e:
                print(f"Error reading {json_path}: {e}")
                continue

        task_instructions = data.get('metadata', {}).get('task_instruction', [])
        question_type = data.get('metadata', {}).get('question_type', "")
        
        for sample in data.get('data', []):
            sample_id = sample.get('sample_id')
            context = sample.get('task_instance', {}).get('context', "")
            
            # 组合问题文本
            instruction_id = sample.get('task_instruction_id', 0)
            prefix = ""
            if task_instructions and instruction_id < len(task_instructions):
                prefix = task_instructions[instruction_id]
            
            question = (prefix + " " + context).strip()
            if not question:
                question = "Please answer the question about this image."
                
            processed_sample = {
                'sample_id': f"{part_name}_{capability_dimension}_{sample_id}",
                'capability_dimension': capability_dimension,
                'part': part_name,
                'question': question,
                'answer': sample.get('response', ""),
                'question_type': question_type,
                'images': []
            }

            # 处理图片路径
            for img_path in sample.get('task_instance', {}).get('images_path', []):
                # 尝试多种可能的路径
                possible_paths = []
                
                # 情况1: 相对于 data.json/images 的简单文件名 (Part2-Part6)
                possible_paths.append(images_dir / img_path)
                
                # 情况2: Part1 的特殊路径 hm3d/xxx 或 scannet/xxx
                if '/' in img_path:
                    parts = img_path.split('/')
                    if len(parts) == 2:
                        prefix_type, scene_id = parts
                        if prefix_type == 'hm3d':
                            possible_paths.append(data_root_path / 'Part1' / 'images' / 'hm3d-v0' / scene_id)
                        elif prefix_type == 'scannet':
                            # 标准化 ScanNet 场景名，移除 002-scannet- 等前缀
                            match = re.search(r'scene\d+_\d+', scene_id)
                            norm_scene_id = match.group(0) if match else scene_id
                            possible_paths.append(data_root_path / 'Part1' / 'images' / 'scannet-v0' / norm_scene_id)
                
                # 检查是否存在
                found_img = False
                for p in possible_paths:
                    # 如果是目录，尝试在目录下找到所有图片
                    if p.exists() and p.is_dir():
                        img_files = sorted(list(p.glob('*.jpg')) + list(p.glob('*.png')))
                        if img_files:
                            if len(img_files) > MAX_IMAGES_PER_SAMPLE:
                                step = (len(img_files) - 1) / (MAX_IMAGES_PER_SAMPLE - 1)
                                indices = [round(i * step) for i in range(MAX_IMAGES_PER_SAMPLE)]
                                img_files = [img_files[i] for i in indices]
                            processed_sample['images'].extend([str(f) for f in img_files])
                            found_img = True
                            break
                    # 如果是文件，直接添加
                    elif p.exists() and p.is_file():
                        processed_sample['images'].append(str(p))
                        found_img = True
                        break
                    # 尝试添加扩展名
                    else:
                        for ext in ['.jpg', '.png']:
                            pext = p.with_suffix(ext)
                            if pext.exists() and pext.is_file():
                                processed_sample['images'].append(str(pext))
                                found_img = True
                                break
                    if found_img: break
                
                if not found_img:
                    # print(f"Warning: Image not found for {img_path} in {json_path}")
                    pass

            if processed_sample['images']:
                all_samples.append(processed_sample)
            
            total_data_count += 1

    print(f"✅ 处理完成！共发现 {total_data_count} 条数据，成功加载 {len(all_samples)} 条带图片的数据。")
    return all_samples

def convert_to_huggingface_format(samples: List[Dict[str, Any]], output_path: str):
    """
    Convert UniEQA data to HuggingFace dataset format storing image paths
    """
    from datasets import Dataset, Features, Value, Sequence

    # 存储图片路径列表，而不是直接存储图片对象，以避免 pyarrow 序列化错误和 OOM
    features = Features({
        'images': Sequence(Value('string')), 
        'question': Value('string'),
        'answer': Value('string'),
        'sample_id': Value('string'),
        'capability_dimension': Value('string'),
        'part': Value('string'),
        'question_type': Value('string')
    })

    hf_data = []
    for sample in tqdm(samples, desc="Preparing data"):
        if sample['images']:
            hf_data.append({
                'images': sample['images'],
                'question': sample['question'],
                'answer': sample['answer'],
                'sample_id': sample['sample_id'],
                'capability_dimension': sample['capability_dimension'],
                'part': sample['part'],
                'question_type': sample.get('question_type', "")
            })

    if hf_data:
        print(f"📦 正在创建包含图片路径的 HuggingFace Dataset (共 {len(hf_data)} 条)...")
        from datasets import DatasetDict
        dataset = Dataset.from_list(hf_data, features=features)
        dataset_dict = DatasetDict({'train': dataset})
        print("💾 正在保存到磁盘...")
        dataset_dict.save_to_disk(output_path)
        print(f"✨ 成功保存全量数据到 {output_path}")
    else:
        print("❌ 没有有效数据可保存")

def main():
    parser = argparse.ArgumentParser(description="UniEQA Dataset Preprocessing Utilities")
    parser.add_argument("--data_root", type=str, required=True, help="Root directory of UniEQA dataset")
    parser.add_argument("--output_path", type=str, help="Output path for processed dataset")
    parser.add_argument("--convert_hf", action="store_true", help="Convert to HuggingFace format")

    args = parser.parse_args()

    if args.convert_hf:
        if not args.output_path:
            print("Error: --output_path required for HuggingFace conversion")
            exit(1)

        samples = load_unieqa_data(args.data_root)
        convert_to_huggingface_format(samples, args.output_path)


    if not args.convert_hf:
        print("UniEQA Dataset Preprocessing Utility")
        print("====================================")
        print()
        print("Usage examples:")
        print("  # Validate dataset")
        print("  python -m embodied_eval.tasks.unieqa.utils.preprocess --data_root /path/to/UniEQA --validate")
        print()
        print("  # Convert to HuggingFace format")
        print("  python -m embodied_eval.tasks.unieqa.utils.preprocess --data_root /path/to/UniEQA --convert_hf --output_path /path/to/output")

if __name__ == "__main__":
    main()