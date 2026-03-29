import os
import json
from pathlib import Path
from tqdm import tqdm

def generate_metadata(root_path):
    root = Path(root_path)
    metadata_path = root / "metadata.jsonl"
    
    # 查找所有的 data.json 文件
    json_files = list(root.glob("**/core/data.json"))
    print(f"Found {len(json_files)} data.json files")
    
    with open(metadata_path, 'w', encoding='utf-8') as f_out:
        for json_path in tqdm(json_files, desc="Processing JSON files"):
            images_dir = json_path.parent / "images"
            # 获取相对于 root 的 images_dir 路径
            rel_images_dir = images_dir.relative_to(root)
            
            try:
                with open(json_path, 'r', encoding='utf-8') as f_in:
                    data_content = json.load(f_in)
            except Exception as e:
                continue
                
            for item in data_content.get('data', []):
                instance = item.get('task_instance', {})
                question = instance.get('context', "")
                answer = item.get('response', item.get('answer', ""))
                img_paths = instance.get('images_path', [])
                
                if not img_paths:
                    continue
                
                # UniEQA Hub structure seems to have one image per row
                img_file = img_paths[0]
                rel_img_path = rel_images_dir / img_file
                
                if not (root / rel_img_path).exists():
                    continue
                
                # Write to metadata.jsonl
                entry = {
                    "file_name": str(rel_img_path),
                    "question": question,
                    "answer": answer,
                    "capability_dimension": json_path.parent.parent.name
                }
                f_out.write(json.dumps(entry, ensure_ascii=False) + "\n")
                
    print(f"Metadata generated at {metadata_path}")

if __name__ == "__main__":
    generate_metadata("/your/path/to/embodied-eval-main/embodied_eval/data/unieqa/UniEQA_Dataset")
