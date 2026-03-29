#!/usr/bin/env python3
"""
Where2Place 坐标缩放逆推工具

功能：对已有的评估结果，尝试不同的坐标缩放方式，找出最佳的缩放配置。

使用方法:
    python reverse_engineer_scaling.py <samples_json_path>
    
或批量分析:
    python reverse_engineer_scaling.py --all
"""

import json
import re
import sys
import argparse
from pathlib import Path
from collections import defaultdict
import numpy as np
from glob import glob


def text2points_original(text, width=640, height=480):
    """原始的坐标解析逻辑（浮点数=归一化，整数=像素）"""
    points = []
    
    # JSON 格式
    json_match = re.search(r"\[\s*\{[\s\S]*?\}\s*\]", text)
    if json_match:
        try:
            data = json.loads(json_match.group())
            for item in data:
                if "point" in item and isinstance(item["point"], list) and len(item["point"]) == 2:
                    x, y = item["point"]
                    x_pixel = int(x * width)
                    y_pixel = int(y * height)
                    points.append((x_pixel, y_pixel))
        except json.JSONDecodeError:
            pass
    
    # 元组格式
    pattern = r"\(([-+]?\d+\.?\d*)\s*,\s*([-+]?\d+\.?\d*)\)"
    matches = re.findall(pattern, text)
    for match in matches:
        x_str, y_str = match
        x = float(x_str) if '.' in x_str else int(x_str)
        y = float(y_str) if '.' in y_str else int(y_str)
        
        is_float = isinstance(x, float) or isinstance(y, float)
        if is_float:
            x = int(x * width)
            y = int(y * height)
        
        points.append((int(x), int(y)))
    
    return np.array(points) if points else np.array([])


def text2points_scale_0_1(text, width=640, height=480):
    """假设：所有坐标都是 0-1 归一化"""
    points = []
    
    # JSON 格式
    json_match = re.search(r"\[\s*\{[\s\S]*?\}\s*\]", text)
    if json_match:
        try:
            data = json.loads(json_match.group())
            for item in data:
                if "point" in item and isinstance(item["point"], list) and len(item["point"]) == 2:
                    x, y = item["point"]
                    x_pixel = int(x * width)
                    y_pixel = int(y * height)
                    points.append((x_pixel, y_pixel))
        except json.JSONDecodeError:
            pass
    
    # 元组/列表格式 - 全部当作归一化
    pattern = r"[\(\[](\d+\.?\d*)\s*,\s*(\d+\.?\d*)[\)\]]"
    matches = re.findall(pattern, text)
    for match in matches:
        x = float(match[0])
        y = float(match[1])
        x_pixel = int(x * width)
        y_pixel = int(y * height)
        points.append((x_pixel, y_pixel))
    
    return np.array(points) if points else np.array([])


def text2points_scale_0_1000(text, width=640, height=480):
    """假设：所有坐标都是 0-1000 归一化"""
    points = []
    
    # Nuoyin 格式
    nuoyin_pattern = r"\[\[(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\]\]"
    nuoyin_matches = re.findall(nuoyin_pattern, text)
    for match in nuoyin_matches:
        x = float(match[0]) / 1000.0
        y = float(match[1]) / 1000.0
        x_pixel = int(x * width)
        y_pixel = int(y * height)
        points.append((x_pixel, y_pixel))
    
    # 元组格式 - 全部除以1000
    pattern = r"\((\d+\.?\d*)\s*,\s*(\d+\.?\d*)\)"
    matches = re.findall(pattern, text)
    for match in matches:
        x = float(match[0]) / 1000.0
        y = float(match[1]) / 1000.0
        x_pixel = int(x * width)
        y_pixel = int(y * height)
        points.append((x_pixel, y_pixel))
    
    return np.array(points) if points else np.array([])


def text2points_scale_percentage(text, width=640, height=480):
    """假设：所有坐标都是 0-100 百分比"""
    points = []
    
    pattern = r"[\(\[](\d+\.?\d*)\s*,\s*(\d+\.?\d*)[\)\]]"
    matches = re.findall(pattern, text)
    for match in matches:
        x = float(match[0]) / 100.0
        y = float(match[1]) / 100.0
        x_pixel = int(x * width)
        y_pixel = int(y * height)
        points.append((x_pixel, y_pixel))
    
    return np.array(points) if points else np.array([])


def text2points_direct_pixel(text, width=640, height=480):
    """假设：所有坐标都是直接的像素坐标"""
    points = []
    
    pattern = r"[\(\[](\d+\.?\d*)\s*,\s*(\d+\.?\d*)[\)\]]"
    matches = re.findall(pattern, text)
    for match in matches:
        x = int(float(match[0]))
        y = int(float(match[1]))
        points.append((x, y))
    
    return np.array(points) if points else np.array([])


def text2points_smart_detect(text, width=640, height=480):
    """智能检测：根据数值大小判断缩放方式"""
    points = []
    
    # JSON 格式
    json_match = re.search(r"\[\s*\{[\s\S]*?\}\s*\]", text)
    if json_match:
        try:
            data = json.loads(json_match.group())
            for item in data:
                if "point" in item and isinstance(item["point"], list) and len(item["point"]) == 2:
                    x, y = item["point"]
                    x_pixel = int(x * width)
                    y_pixel = int(y * height)
                    points.append((x_pixel, y_pixel))
        except json.JSONDecodeError:
            pass
    
    # 元组格式 - 智能判断
    pattern = r"[\(\[](\d+\.?\d*)\s*,\s*(\d+\.?\d*)[\)\]]"
    matches = re.findall(pattern, text)
    for match in matches:
        x_val = float(match[0])
        y_val = float(match[1])
        
        # 判断缩放方式
        if x_val <= 1.0 and y_val <= 1.0:
            # 0-1 归一化
            x = int(x_val * width)
            y = int(y_val * height)
        elif x_val <= 100 and y_val <= 100:
            # 可能是百分比
            x = int(x_val / 100.0 * width)
            y = int(y_val / 100.0 * height)
        elif x_val <= 1000 and y_val <= 1000:
            # 可能是 0-1000
            x = int(x_val / 1000.0 * width)
            y = int(y_val / 1000.0 * height)
        else:
            # 直接像素
            x = int(x_val)
            y = int(y_val)
        
        points.append((x, y))
    
    return np.array(points) if points else np.array([])


def spatial_reference(points, mask, width=640, height=480, threshold=0.5):
    """计算空间参考准确率"""
    try:
        if len(points) == 0:
            return 0.0
        
        # 确保 mask 是二值化的
        if mask.dtype != bool:
            mask = mask > threshold
        
        # 检查点是否在范围内并且在mask中
        in_range = (points[:, 0] >= 0) & (points[:, 0] < mask.shape[1]) \
                    & (points[:, 1] >= 0) & (points[:, 1] < mask.shape[0])
        
        if not in_range.any():
            return 0.0
        
        # 计算准确率
        valid_points = points[in_range]
        hits = mask[valid_points[:, 1], valid_points[:, 0]]
        
        # 计算命中率
        acc = np.concatenate([
            hits,
            np.zeros(points.shape[0] - in_range.sum())
        ]).mean()
        
        return float(acc)
    except Exception as e:
        return 0.0


def analyze_sample_with_scales(sample, width=640, height=480):
    """使用不同的缩放策略分析单个样本"""
    
    # 获取预测和目标
    pred = sample.get('prediction') or (sample.get('resps', [['']])[0][0] if sample.get('resps') else '')
    
    # 获取目标mask
    if 'doc' in sample and 'mask' in sample['doc']:
        mask = np.array(sample['doc']['mask'])
    elif 'target' in sample:
        target = sample['target']
        if isinstance(target, list) and len(target) == 4:
            # bbox格式
            x0, y0, x1, y1 = target
            mask = np.zeros((height, width), dtype=bool)
            mask[y0:y1, x0:x1] = 1
        else:
            return None
    else:
        return None
    
    # 归一化mask
    if mask.dtype != bool:
        mask = (mask / 255.0) > 0.5
    
    # 尝试不同的缩放策略
    strategies = {
        'original': text2points_original,
        '0-1_normalized': text2points_scale_0_1,
        '0-1000_normalized': text2points_scale_0_1000,
        '0-100_percentage': text2points_scale_percentage,
        'direct_pixel': text2points_direct_pixel,
        'smart_detect': text2points_smart_detect,
    }
    
    results = {}
    for strategy_name, strategy_func in strategies.items():
        try:
            points = strategy_func(pred, width, height)
            acc = spatial_reference(points, mask, width, height)
            results[strategy_name] = {
                'accuracy': acc,
                'num_points': len(points),
                'points_sample': points[:3].tolist() if len(points) > 0 else []
            }
        except Exception as e:
            results[strategy_name] = {
                'accuracy': 0.0,
                'num_points': 0,
                'points_sample': [],
                'error': str(e)
            }
    
    return results


def analyze_model_logs(samples_path, max_samples=None):
    """分析单个模型的日志"""
    
    print(f"\n{'='*80}")
    print(f"分析文件: {samples_path}")
    print(f"{'='*80}\n")
    
    # 读取样本
    samples = []
    try:
        with open(samples_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    samples.append(json.loads(line))
    except Exception as e:
        print(f"❌ 读取文件失败: {e}")
        return None
    
    if not samples:
        print("❌ 文件为空")
        return None
    
    # 限制样本数量
    if max_samples:
        samples = samples[:max_samples]
    
    print(f"✅ 读取 {len(samples)} 个样本\n")
    
    # 统计每种策略的准确率
    strategy_stats = defaultdict(lambda: {'total_acc': 0.0, 'count': 0, 'sample_accs': []})
    
    for idx, sample in enumerate(samples):
        results = analyze_sample_with_scales(sample)
        if results:
            for strategy_name, result in results.items():
                strategy_stats[strategy_name]['total_acc'] += result['accuracy']
                strategy_stats[strategy_name]['count'] += 1
                strategy_stats[strategy_name]['sample_accs'].append(result['accuracy'])
                
                # 记录第一个样本的详细信息
                if idx == 0:
                    strategy_stats[strategy_name]['first_sample'] = result
    
    # 计算平均准确率
    results_summary = {}
    for strategy_name, stats in strategy_stats.items():
        if stats['count'] > 0:
            avg_acc = stats['total_acc'] / stats['count']
            results_summary[strategy_name] = {
                'average_accuracy': avg_acc,
                'count': stats['count'],
                'first_sample': stats.get('first_sample', {}),
                'std': np.std(stats['sample_accs']) if stats['sample_accs'] else 0.0
            }
    
    # 按准确率排序
    sorted_results = sorted(results_summary.items(), key=lambda x: x[1]['average_accuracy'], reverse=True)
    
    # 打印结果
    print("📊 不同缩放策略的准确率对比")
    print("-" * 80)
    print(f"{'策略':<25} {'准确率':<12} {'标准差':<10} {'第一个样本点数'}")
    print("-" * 80)
    
    for strategy_name, stats in sorted_results:
        print(f"{strategy_name:<25} {stats['average_accuracy']:<12.4f} {stats['std']:<10.4f} "
              f"{stats['first_sample'].get('num_points', 0)}")
    
    # 显示最佳策略
    if sorted_results:
        best_strategy, best_stats = sorted_results[0]
        print(f"\n✨ 推荐策略: {best_strategy}")
        print(f"   准确率: {best_stats['average_accuracy']:.4f}")
        
        # 给出配置建议
        if best_strategy == '0-1000_normalized':
            print(f"\n💡 配置建议:")
            print(f"   export Nuoyin_API_BASE=\"http://your-api-base\"")
        elif best_strategy == 'smart_detect':
            print(f"\n💡 配置建议:")
            print(f"   export NORMALIZE_PIXEL_COORDS=true")
        elif best_strategy == '0-1_normalized':
            print(f"\n💡 配置建议:")
            print(f"   使用默认配置（所有坐标强制归一化）")
        elif best_strategy == 'direct_pixel':
            print(f"\n💡 配置建议:")
            print(f"   模型输出像素坐标，需要修改prompt或添加后处理")
        elif best_strategy == 'original':
            print(f"\n💡 配置建议:")
            print(f"   使用当前默认配置即可")
        
        # 显示第一个样本的坐标
        first_sample = best_stats.get('first_sample', {})
        if first_sample.get('points_sample'):
            print(f"\n📍 第一个样本的坐标示例:")
            print(f"   {first_sample['points_sample'][:3]}")
    
    return results_summary


def batch_analyze_all_models(logs_dir):
    """批量分析所有模型"""
    
    logs_path = Path(logs_dir)
    if not logs_path.exists():
        print(f"❌ 日志目录不存在: {logs_dir}")
        return
    
    # 查找所有样本文件
    sample_files = list(logs_path.rglob("samples_*.json"))
    
    if not sample_files:
        print(f"❌ 未找到任何样本文件")
        return
    
    print(f"\n找到 {len(sample_files)} 个样本文件\n")
    
    # 分析每个文件
    all_results = {}
    for sample_file in sorted(sample_files):
        model_name = sample_file.parent.parent.name
        timestamp = sample_file.parent.name
        key = f"{model_name}/{timestamp}"
        
        results = analyze_model_logs(sample_file, max_samples=100)
        if results:
            all_results[key] = results
    
    # 生成汇总报告
    print(f"\n\n{'='*80}")
    print("📋 所有模型的最佳策略汇总")
    print(f"{'='*80}\n")
    
    print(f"{'模型':<40} {'最佳策略':<25} {'准确率':<10}")
    print("-" * 80)
    
    for model_key, results in sorted(all_results.items()):
        sorted_results = sorted(results.items(), key=lambda x: x[1]['average_accuracy'], reverse=True)
        if sorted_results:
            best_strategy, best_stats = sorted_results[0]
            print(f"{model_key:<40} {best_strategy:<25} {best_stats['average_accuracy']:<10.4f}")
    
    # 保存详细报告
    report_path = logs_path / "scaling_analysis_report.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ 详细报告已保存到: {report_path}")


def main():
    parser = argparse.ArgumentParser(description='Where2Place 坐标缩放逆推工具')
    parser.add_argument('samples_path', nargs='?', help='样本文件路径')
    parser.add_argument('--all', action='store_true', help='分析所有模型')
    parser.add_argument('--max-samples', type=int, default=None, help='最大样本数量（默认全部）')
    parser.add_argument('--logs-dir', default='/your/path/to/embodied-eval-main/logs/where2place',
                        help='日志目录路径')
    
    args = parser.parse_args()
    
    if args.all:
        batch_analyze_all_models(args.logs_dir)
    elif args.samples_path:
        samples_path = Path(args.samples_path)
        if not samples_path.exists():
            # 尝试glob匹配
            matches = glob(str(samples_path))
            if matches:
                samples_path = Path(matches[0])
            else:
                print(f"❌ 文件不存在: {samples_path}")
                sys.exit(1)
        
        analyze_model_logs(samples_path, max_samples=args.max_samples)
    else:
        parser.print_help()
        print("\n示例:")
        print("  # 分析单个模型")
        print("  python reverse_engineer_scaling.py logs/where2place/qwen3_vl/*/samples_*.json")
        print("\n  # 批量分析所有模型")
        print("  python reverse_engineer_scaling.py --all")


if __name__ == '__main__':
    main()





