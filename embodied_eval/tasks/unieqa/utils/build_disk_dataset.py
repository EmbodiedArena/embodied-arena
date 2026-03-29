import os
import json
from datasets import Dataset, DatasetDict, Features, Image, Value, Sequence
from pathlib import Path
from tqdm import tqdm

def build_dataset(root_path):
    root = Path(root_path)
    all_data = []
    
    # 1. 递归扫描所有包含 data.json 的目录
    json_files = list(root.glob("**/data.json"))
    print(f"🚀 发现 {len(json_files)} 个配置文件，开始全量整合...")

    for json_path in tqdm(json_files, desc="Parsing"):
        # 探测图片目录
        # 逻辑：优先尝试 json 同级的 images 目录，或者 core/images
        images_dir = json_path.parent / "images"
        if not images_dir.exists():
            images_dir = json_path.parent / "core" / "images"
        
        # 特殊处理：如果 json 在 data/PartX/... 下，图片在 data/PartX/images/ 下
        if not images_dir.exists():
            parts = list(json_path.parts)
            for part in parts:
                if part.startswith("Part") and part[4:].isdigit():
                    # 找到 data 目录的位置
                    try:
                        part_idx = parts.index(part)
                        # images 目录在 data/PartX/images
                        images_dir = Path(*parts[:part_idx+1]) / "images"
                    except:
                        pass
                    break
        
        # 维度名
        dim = json_path.parent.name
        if dim == "core":
            dim = json_path.parent.parent.name

        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data_content = json.load(f)
        except:
            continue
            
        for item in data_content.get('data', []):
            instance = item.get('task_instance', {})
            question = instance.get('context', "")
            answer = item.get('response', item.get('answer', ""))
            img_paths = instance.get('images_path', [])
            
            if not img_paths:
                continue
                
            # 处理多图（视频帧）
            valid_abs_paths = []
            for img_file in img_paths:
                abs_img_path = images_dir / img_file
                
                # 处理路径不匹配：data.json 中是 hm3d/ 但实际目录是 hm3d-v0/
                if not abs_img_path.exists():
                    img_file_fixed = img_file.replace("hm3d/", "hm3d-v0/").replace("scannet/", "scannet-v0/")
                    if img_file_fixed != img_file:
                        abs_img_path = images_dir / img_file_fixed
                
                # 如果路径是目录，找目录中的第一张图片
                if abs_img_path.exists() and abs_img_path.is_dir():
                    img_files = list(abs_img_path.glob("*.png")) + list(abs_img_path.glob("*.jpg"))
                    if img_files:
                        img_files.sort()
                        abs_img_path = img_files[0]
                
                if abs_img_path.exists() and abs_img_path.is_file():
                    valid_abs_paths.append(str(abs_img_path))
            
            if valid_abs_paths:
                all_data.append({
                    "image": valid_abs_paths[0],  # 保持单图字段兼容旧代码
                    "images": valid_abs_paths,    # 新增多图字段支持视频帧
                    "question": question,
                    "answer": answer,
                    "capability_dimension": dim,
                    "sample_id": item.get('sample_id', 0)
                })
            
    print(f"✅ 整合完成！共收集到 {len(all_data)} 条有效样本。")
    
    features = Features({
        "image": Image(),
        "images": Sequence(Image()),
        "question": Value("string"),
        "answer": Value("string"),
        "capability_dimension": Value("string"),
        "sample_id": Value("int32")
    })
    
    raw_dataset = Dataset.from_list(all_data, features=features)
    return DatasetDict({"train": raw_dataset})
    
    raw_dataset = Dataset.from_list(all_data, features=features)
    return DatasetDict({"train": raw_dataset})

if __name__ == "__main__":
    root_path = "/your/path/to/embodied-eval-main/embodied_eval/data/unieqa/111/UniEQA"
    output_disk_path = "/your/path/to/embodied-eval-main/embodied_eval/data/unieqa/UniEQA_Dataset_Disk"
    
    ds_dict = build_dataset(root_path)
    ds_dict.save_to_disk(output_disk_path)
    print(f"🚀 完整数据集已无损保存至: {output_disk_path}")
