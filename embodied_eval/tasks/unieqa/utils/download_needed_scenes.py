#!/usr/bin/env python3
"""
只下载 UniEQA 数据中需要的 ScanNet 场景的 .sens 文件
"""
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Set


def get_needed_scannet_scenes(data_dir: Path) -> Set[str]:
    """从 UniEQA 数据中提取需要的 ScanNet 场景ID"""
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
                                match = re.search(r'scene\d{4}_\d{2}', scene_id, re.IGNORECASE)
                                if match:
                                    actual_scene_id = match.group(0)
                                    needed_scenes.add(actual_scene_id)
                        break
        except Exception as e:
            print(f"   ⚠️ 警告: 读取 {data_file} 时出错: {e}")
            continue
    
    return needed_scenes


def check_downloaded_scenes(scannet_root: Path) -> Set[str]:
    """检查已下载的场景（必须是非空文件）"""
    downloaded = set()
    
    for folder in ["scans", "scans_test"]:
        scans_dir = scannet_root / folder
        if scans_dir.exists():
            for sens_file in scans_dir.rglob('*.sens'):
                # 只有大于 1KB 的文件才认为是有效的（排除空文件或损坏的小文件）
                if sens_file.stat().st_size > 1024:
                    scene_id = sens_file.stem
                    downloaded.add(scene_id)
    
    return downloaded


def download_scene(scannet_root: Path, scene_id: str, script_path: Path):
    """下载单个场景"""
    # 预检查：如果存在 0 字节的文件，先删除它，否则 download_scannet.py 会跳过
    for folder in ["scans", "scans_test"]:
        sens_file = scannet_root / folder / scene_id / f"{scene_id}.sens"
        if sens_file.exists() and sens_file.stat().st_size < 1024:
            print(f"  ⚠️  发现损坏的文件 {sens_file.name}，正在删除以重新下载...")
            sens_file.unlink()

    # download_scannet.py 需要两次确认：
    # 1. 确认条款（按任意键）
    # 2. 确认下载 .sens 文件（按回车，不是 'n'）
    result = subprocess.run(
        ["python", str(script_path), "-o", str(scannet_root), "--type", ".sens", "--id", scene_id],
        input="\n\n",  # 第一次确认条款，第二次确认下载 .sens（按回车）
        text=True,
        capture_output=True
    )
    
    # 检查是否成功
    if result.returncode == 0:
        # 检查输出确认下载成功或已存在
        if "Downloaded scan" in result.stdout or "WARNING: skipping download" in result.stdout:
            if "WARNING: skipping download" in result.stdout:
                print(f"  ✓ {scene_id} (已存在，跳过)")
            else:
                print(f"  ✓ {scene_id} (下载成功)")
            return True
    
    # 下载失败
    print(f"  ✗ {scene_id} (下载失败)")
    if result.stderr:
        print(f"    Error: {result.stderr[:200]}")  # 只显示前200个字符
    return False


def main():
    if len(sys.argv) < 3:
        print("Usage: python download_needed_scenes.py <scannet_root> <unieqa_data_dir>")
        sys.exit(1)
    
    scannet_root = Path(sys.argv[1])
    unieqa_data_dir = Path(sys.argv[2])
    download_script = scannet_root / "download_scannet.py"
    
    if not download_script.exists():
        print(f"Error: download_scannet.py not found at {download_script}")
        sys.exit(1)
    
    print(f"📁 UniEQA 数据目录: {unieqa_data_dir}")
    print(f"   目录存在: {unieqa_data_dir.exists()}")
    
    print("🔍 检查需要的场景...")
    needed_scenes = get_needed_scannet_scenes(unieqa_data_dir)
    print(f"   需要下载的场景数: {len(needed_scenes)}")
    if needed_scenes:
        print(f"   场景列表: {sorted(list(needed_scenes))[:5]}..." if len(needed_scenes) > 5 else f"   场景列表: {sorted(list(needed_scenes))}")
    
    print("🔍 检查已下载的场景...")
    downloaded_scenes = check_downloaded_scenes(scannet_root)
    print(f"   已下载的场景数: {len(downloaded_scenes)}")
    
    missing_scenes = needed_scenes - downloaded_scenes
    
    if not missing_scenes:
        print("\n✅ 所有需要的场景都已下载！无需下载。")
        return 0
    
    print(f"\n📥 需要下载 {len(missing_scenes)} 个缺失的场景:")
    for i, scene_id in enumerate(sorted(missing_scenes), 1):
        print(f"  [{i}/{len(missing_scenes)}] {scene_id}")
    
    print(f"\n🚀 开始下载缺失的场景...")
    success_count = 0
    failed_scenes = []
    
    for i, scene_id in enumerate(sorted(missing_scenes), 1):
        print(f"[{i}/{len(missing_scenes)}] 下载 {scene_id}...", end=" ")
        if download_scene(scannet_root, scene_id, download_script):
            success_count += 1
        else:
            failed_scenes.append(scene_id)
    
    print(f"\n{'='*60}")
    print(f"✅ 下载完成！")
    print(f"   成功: {success_count}/{len(missing_scenes)} 个场景")
    if failed_scenes:
        print(f"   失败: {len(failed_scenes)} 个场景")
        print(f"   失败的场景: {', '.join(failed_scenes)}")
    print(f"{'='*60}")
    
    return 0 if not failed_scenes else 1


if __name__ == "__main__":
    sys.exit(main())
