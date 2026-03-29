"""
重新计算准确率脚本
用于处理百分比格式坐标的样本文件
"""
import json
import numpy as np
import re
import os
from collections import defaultdict


def text2points_percentage(text, width=640, height=480):
    """
    解析百分比格式的坐标点（0-100范围）
    
    Args:
        text: 包含坐标的文本
        width: 图像宽度
        height: 图像高度
    
    Returns:
        numpy array of shape (N, 2) containing pixel coordinates
    """
    points = []
    
    # 解析元组格式: (x, y)
    pattern = r"\(([-+]?\d+\.?\d*(?:,\s*[-+]?\d+\.?\d*)*?)\)"
    matches = re.findall(pattern, text)
    
    for match in matches:
        vector = [float(num) if '.' in num else int(num) for num in match.split(',')]
        
        if len(vector) == 2:
            x, y = vector
            # 百分比格式：0-100 -> 像素坐标
            x_pixel = int(x / 100.0 * width)
            y_pixel = int(y / 100.0 * height)
            points.append((x_pixel, y_pixel))
        elif len(vector) == 4:
            # bbox格式
            x0, y0, x1, y1 = vector
            # 百分比格式转像素
            x0 = int(x0 / 100.0 * width)
            y0 = int(y0 / 100.0 * height)
            x1 = int(x1 / 100.0 * width)
            y1 = int(y1 / 100.0 * height)
            
            # 生成bbox内的所有点
            mask = np.zeros((height, width), dtype=bool)
            mask[int(y0):int(y1), int(x0):int(x1)] = 1
            y_coords, x_coords = np.where(mask)
            points.extend(list(np.stack([x_coords, y_coords], axis=1)))
    
    return np.array(points) if points else np.array([]).reshape(0, 2)


def spatial_reference(pred, target, width=640, height=480, threshold=0.5):
    """
    计算空间参考准确率
    
    Args:
        pred: 预测的文本（包含坐标）
        target: 目标区域（bbox格式 [x0, y0, x1, y1]）
        width: 图像宽度
        height: 图像高度
        threshold: mask阈值
    
    Returns:
        accuracy: 准确率
    """
    try:
        # 解析预测的点
        points = text2points_percentage(pred.strip(), width=width, height=height)
        
        # 处理target（bbox格式）
        if isinstance(target, list) and len(target) == 4:
            x0, y0, x1, y1 = target
            binary_mask = np.zeros((height, width), dtype=bool)
            binary_mask[y0:y1, x0:x1] = 1
            mask = binary_mask
        elif isinstance(target, np.ndarray):
            if target.dtype != bool:
                mask = target > threshold
            else:
                mask = target
        else:
            return 0.0
        
        # 如果没有预测点，返回0
        if len(points) == 0:
            return 0.0
        
        # 检查点是否在有效范围内
        in_range = (points[:, 0] >= 0) & (points[:, 0] < mask.shape[1]) \
                    & (points[:, 1] >= 0) & (points[:, 1] < mask.shape[0])
        
        # 计算准确率：有效点中在mask内的比例
        if in_range.sum() > 0:
            valid_points = points[in_range]
            acc = mask[valid_points[:, 1], valid_points[:, 0]].mean()
        else:
            # 所有点都超出范围
            acc = 0.0
        
        return float(acc)
    except Exception as e:
        print(f"Error in spatial_reference: {e}")
        return 0.0


def recompute_accuracy(sample_file_path, results_file_path=None):
    """
    重新计算样本文件中的准确率
    
    Args:
        sample_file_path: 样本文件路径
        results_file_path: 结果文件路径（可选）
    """
    # 读取样本数据
    print(f"读取样本文件: {sample_file_path}")
    with open(sample_file_path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    
    # 统计信息
    type_correct = defaultdict(float)
    type_total = defaultdict(int)
    
    # 重新计算每个样本的准确率
    for i, doc in enumerate(data):
        # 获取预测结果
        pred_raw = doc["resps"][0][0] if doc["resps"] and doc["resps"][0] else ""
        # 获取目标区域
        target = doc["target"]
        
        # 计算准确率
        acc = spatial_reference(pred_raw, target, width=640, height=480)
        
        # 更新准确率
        old_acc = doc.get("accuracy", 0.0)
        doc["accuracy"] = acc
        
        # 统计
        qtype = doc.get("question_type", "unknown")
        type_correct[qtype] += acc
        type_total[qtype] += 1
        
        # 如果准确率有变化，打印信息
        if abs(acc - old_acc) > 0.001:
            print(f"Doc {i} (doc_id={doc.get('doc_id', 'unknown')}): "
                  f"accuracy {old_acc:.4f} -> {acc:.4f}")
    
    # 保存更新后的样本文件
    print(f"\n保存更新后的样本文件: {sample_file_path}")
    with open(sample_file_path, "w", encoding="utf-8") as f:
        for doc in data:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
    
    # 计算各类型的成功率
    type_success_rate = {
        f"{qtype}_accuracy": round(type_correct[qtype] / type_total[qtype], 4)
        for qtype in type_total
    }
    
    # 计算总体准确率
    values = list(type_success_rate.values())
    overall = round(sum(values) / len(values), 4) if values else 0.0
    type_success_rate["overall"] = overall
    
    # 打印结果
    print("\n" + "="*60)
    print("重新计算后的结果：")
    print("="*60)
    for key, value in sorted(type_success_rate.items()):
        print(f"{key}: {value}")
    print("="*60)
    
    # 保存结果文件
    if results_file_path is None:
        results_file_path = sample_file_path.replace("samples_", "results_").replace(".json", "_recomputed.json")
    
    print(f"\n保存结果文件: {results_file_path}")
    with open(results_file_path, "w", encoding="utf-8") as f:
        json.dump(type_success_rate, f, ensure_ascii=False, indent=2)
    
    return type_success_rate


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("用法: python recompute_accuracy.py <sample_file_path> [results_file_path]")
        print("\n示例:")
        print("  python recompute_accuracy.py samples_where2place-point.json")
        print("  python recompute_accuracy.py samples_where2place-point.json results_where2place-point.json")
        sys.exit(1)
    
    sample_file = sys.argv[1]
    results_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    recompute_accuracy(sample_file, results_file)




