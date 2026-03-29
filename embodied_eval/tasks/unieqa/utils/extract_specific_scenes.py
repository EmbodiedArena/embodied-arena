#!/usr/bin/env python3
"""
提取指定的 ScanNet 场景的图片帧
"""
import argparse
from pathlib import Path
from typing import List, Set
import tqdm
from SensorData import SensorData


def get_needed_scannet_scenes(data_dir: Path) -> Set[str]:
    """从 UniEQA 数据中提取需要的 ScanNet 场景ID"""
    import json
    import re
    
    needed_scenes = set()
    
    part1_dir = data_dir / 'Part1'
    if not part1_dir.exists():
        print(f"   ⚠️ 警告: Part1 目录不存在: {part1_dir}")
        return needed_scenes
    
    data_files = list((part1_dir).rglob('**/data.json'))
    print(f"   找到 {len(data_files)} 个 data.json 文件")
    
    for data_file in sorted(data_files):
        try:
            with open(data_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for item in data.get('data', []):
                images_path = item.get('task_instance', {}).get('images_path', [])
                
                for img_path in images_path:
                    if 'scannet' in img_path.lower():
                        parts = img_path.split('/')
                        if len(parts) == 2:
                            prefix, scene_id = parts
                            if prefix == 'scannet':
                                # 提取实际的 ScanNet 场景ID（例如：002-scannet-scene0709_00 -> scene0709_00）
                                match = re.search(r'scene\d{4}_\d{2}', scene_id, re.IGNORECASE)
                                if match:
                                    actual_scene_id = match.group(0)
                                    needed_scenes.add(actual_scene_id)
                        break
        except Exception as e:
            print(f"   ⚠️ 警告: 读取 {data_file} 时出错: {e}")
            continue
    
    return needed_scenes


def get_scene_path(scannet_root: Path, scene_id: str) -> Path:
    """查找场景的 .sens 文件路径"""
    for folder in ["scans", "scans_test"]:
        scene_path = scannet_root / folder / scene_id / (scene_id + ".sens")
        if scene_path.exists():
            return scene_path
    raise ValueError(f"Scene {scene_id} not found in {scannet_root}")


def extract_frames(
    scene_path: Path, 
    output_folder: Path, 
    rgb_only: bool = True,
    max_num_frames: int = 600
) -> None:
    """提取单个场景的图片帧"""
    output_folder.mkdir(exist_ok=True, parents=True)
    output_folder = str(output_folder)

    print(f"Extracting frames to: {output_folder}")
    try:
        sd = SensorData(str(scene_path))
    except Exception as e:
        print(f"Failed to load SensorData for {scene_path}: {e}")
        return

    if not rgb_only:
        sd.export_intrinsics(output_folder)
        sd.export_poses(output_folder, num_frames=max_num_frames)
        sd.export_depth_images(output_folder, num_frames=max_num_frames)
    sd.export_color_images(output_folder, num_frames=max_num_frames)
    print(f"Extracting frames to: {output_folder} done!")


def main():
    parser = argparse.ArgumentParser(description='Extract frames from specific ScanNet scenes')
    parser.add_argument(
        "--scannet-root",
        type=Path,
        default=".",
        help="path to scannet data root (default: current directory)",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        required=True,
        help="path to output folder",
    )
    parser.add_argument(
        "--unieqa-data-dir",
        type=Path,
        help="path to UniEQA data directory (to auto-detect needed scenes)",
    )
    parser.add_argument(
        "--scenes",
        type=str,
        nargs="+",
        help="specific scene IDs to extract (e.g., scene0709_00 scene0714_00)",
    )
    parser.add_argument(
        "--rgb-only",
        action="store_true",
        help="only extract rgb frames (default: false)",
    )
    parser.add_argument(
        "--max-num-frames",
        type=int,
        default=600,
        help="maximum frames to extract from a scene (default: 600)",
    )
    args = parser.parse_args()
    
    args.output_directory.mkdir(parents=True, exist_ok=True)
    
    # 确定要提取的场景列表
    if args.scenes:
        scenes_to_extract = set(args.scenes)
        print(f"Extracting {len(scenes_to_extract)} specified scenes")
    elif args.unieqa_data_dir:
        print(f"📁 UniEQA 数据目录: {args.unieqa_data_dir}")
        print(f"   目录存在: {args.unieqa_data_dir.exists()}")
        scenes_to_extract = get_needed_scannet_scenes(args.unieqa_data_dir)
        print(f"Auto-detected {len(scenes_to_extract)} scenes from UniEQA data")
        if scenes_to_extract:
            print(f"   场景列表: {sorted(list(scenes_to_extract))[:10]}..." if len(scenes_to_extract) > 10 else f"   场景列表: {sorted(list(scenes_to_extract))}")
    else:
        print("Error: Either --scenes or --unieqa-data-dir must be specified")
        return
    
    if not scenes_to_extract:
        print("⚠️  警告: 没有找到需要提取的场景！")
        return
    
    print(f"Scenes to extract: {sorted(scenes_to_extract)}")
    
    # 提取每个场景
    for scene_id in tqdm.tqdm(sorted(scenes_to_extract), desc="Extracting scenes"):
        try:
            scene_path = get_scene_path(args.scannet_root, scene_id)
            output_folder = args.output_directory / scene_id
            
            # 检查 .sens 文件是否有效
            if scene_path.stat().st_size < 1024:
                print(f"\nWarning: Skipping {scene_id}, .sens file is empty or too small.")
                continue
                
            extract_frames(
                scene_path=scene_path,
                output_folder=output_folder,
                rgb_only=args.rgb_only,
                max_num_frames=args.max_num_frames
            )
        except ValueError as e:
            print(f"\nWarning: {e}")
            continue
        except Exception as e:
            print(f"\nError processing {scene_id}: {e}")
            continue
    
    print(f"\n✅ Extraction complete! Extracted {len(scenes_to_extract)} scenes to {args.output_directory}")


if __name__ == "__main__":
    main()
