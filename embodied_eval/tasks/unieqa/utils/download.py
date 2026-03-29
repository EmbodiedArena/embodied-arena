import os
import json
import subprocess
import argparse
from pathlib import Path
from tqdm import tqdm

def download_unieqa_dataset(output_path: str):
    root = Path(output_path)
    base_url = "https://huggingface.co/datasets/TJURL-Lab/UniEQA/resolve/main"
    
    # 1. 递归找到所有的 data.json 文件
    json_files = list(root.glob("**/core/data.json"))
    print(f"🚀 发现 {len(json_files)} 个配置文件，开始检查图片...")

    total_downloaded = 0
    
    for json_path in json_files:
        images_dir = json_path.parent / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        
        # 获取相对于数据集根目录的路径，用于拼接 URL
        try:
            rel_path = json_path.parent.relative_to(root)
        except:
            continue
            
        with open(json_path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
            except:
                continue
        
        image_files = set()
        for s in data.get('data', []):
            for img_path in s.get('task_instance', {}).get('images_path', []):
                image_files.add(img_path)
        
        if not image_files:
            continue

        print(f"📦 处理目录: {rel_path} (需检查 {len(image_files)} 张图)")
        
        downloaded = 0
        for img_file in image_files:
            img_dst = images_dir / img_file
            # 检查是否不存在或者是 LFS 指针 (小于 500 字节)
            if not img_dst.exists() or img_dst.stat().st_size < 500:
                img_url = f"{base_url}/{rel_path}/images/{img_file}"
                try:
                    subprocess.run(["wget", "-q", "-O", str(img_dst), img_url], check=True, timeout=15)
                    downloaded += 1
                except:
                    pass
        
        print(f"  ✓ 已补全 {downloaded} 张新图片")
        total_downloaded += downloaded

    print(f"✨ 补全任务完成！共下载了 {total_downloaded} 张图片。")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--auto_download", action="store_true")
    args = parser.parse_args()
    
    if args.auto_download:
        download_unieqa_dataset(args.output_path)
