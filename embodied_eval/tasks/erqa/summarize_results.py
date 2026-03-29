#!/usr/bin/env python3
"""
汇总所有模型的ERQA评估结果
"""

import json
from pathlib import Path
from collections import defaultdict

def get_latest_result(model_dir):
    """获取模型目录下最新的结果文件"""
    result_files = list(model_dir.rglob('results_erqa.json'))
    
    if not result_files:
        return None
    
    # 按时间戳排序，取最新的
    result_files.sort(key=lambda x: x.parent.name, reverse=True)
    latest_file = result_files[0]
    
    with open(latest_file, 'r', encoding='utf-8') as f:
        result = json.load(f)
    
    # 获取对应的样本文件信息
    sample_file = latest_file.parent / 'samples_erqa.json'
    if sample_file.exists():
        with open(sample_file, 'r', encoding='utf-8') as f:
            sample_count = sum(1 for _ in f)
    else:
        sample_count = 0
    
    return {
        'timestamp': latest_file.parent.name,
        'result_file': str(latest_file),
        'sample_count': sample_count,
        'results': result
    }

def main():
    logs_dir = Path('/your/path/to/embodied-eval-main/logs/erqa')
    
    all_results = {}
    
    print("汇总ERQA评估结果...\n")
    
    # 遍历所有模型
    for model_dir in sorted(logs_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        
        model_name = model_dir.name
        print(f"处理模型: {model_name}")
        
        latest_result = get_latest_result(model_dir)
        
        if latest_result:
            all_results[model_name] = latest_result
            overall = latest_result['results'].get('overall', 0)
            accuracy_avg = latest_result['results'].get('accuracy_average', 0)
            print(f"  时间戳: {latest_result['timestamp']}")
            print(f"  样本数: {latest_result['sample_count']}")
            print(f"  Overall: {overall:.4f}")
            print(f"  Accuracy Average: {accuracy_avg:.4f}")
        else:
            print(f"  未找到结果文件")
        print()
    
    # 保存汇总结果
    output_file = logs_dir / 'all_models_results.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 汇总结果已保存到: {output_file}")
    
    # 创建一个排行榜
    leaderboard = []
    for model_name, data in all_results.items():
        leaderboard.append({
            'model': model_name,
            'overall_accuracy': data['results'].get('overall', 0),
            'accuracy_average': data['results'].get('accuracy_average', 0),
            'timestamp': data['timestamp'],
            'sample_count': data['sample_count']
        })
    
    # 按overall_accuracy降序排序
    leaderboard.sort(key=lambda x: x['overall_accuracy'], reverse=True)
    
    # 保存排行榜
    leaderboard_file = logs_dir / 'erqa_leaderboard.json'
    with open(leaderboard_file, 'w', encoding='utf-8') as f:
        json.dump(leaderboard, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 排行榜已保存到: {leaderboard_file}")
    
    # 打印排行榜
    print("\n" + "="*80)
    print("ERQA 模型排行榜")
    print("="*80)
    print(f"{'排名':<6} {'模型':<30} {'Overall':<12} {'样本数':<10} {'时间戳'}")
    print("-"*80)
    
    for i, item in enumerate(leaderboard, 1):
        print(f"{i:<6} {item['model']:<30} {item['overall_accuracy']:<12.4f} {item['sample_count']:<10} {item['timestamp']}")
    
    print("="*80)

if __name__ == '__main__':
    main()



