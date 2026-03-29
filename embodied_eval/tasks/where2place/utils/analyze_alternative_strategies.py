#!/usr/bin/env python3
"""
分析除了 original 和 0-1_normalized 之外的最佳策略
"""

import json
from pathlib import Path

# 读取之前的分析结果
report_path = Path("/your/path/to/embodied-eval-main/logs/where2place/scaling_analysis_report.json")

if not report_path.exists():
    print("❌ 请先运行 utils/reverse_engineer_scaling.py --all")
    exit(1)

with open(report_path, 'r') as f:
    all_results = json.load(f)

print("="*80)
print("📊 排除 original 和 0-1_normalized 后的最佳策略")
print("="*80)
print()

# 要排除的策略
excluded_strategies = ['original', '0-1_normalized']

print(f"{'模型':<45} {'最佳策略':<25} {'准确率':<10} {'原始准确率':<12}")
print("-"*80)

summary = []

for model_key, results in sorted(all_results.items()):
    # 过滤掉排除的策略
    filtered_results = {
        strategy: stats 
        for strategy, stats in results.items() 
        if strategy not in excluded_strategies
    }
    
    if not filtered_results:
        continue
    
    # 找出原始最佳策略的准确率
    original_acc = results.get('original', {}).get('average_accuracy', 0.0)
    
    # 排序找出最佳策略
    sorted_results = sorted(
        filtered_results.items(), 
        key=lambda x: x[1]['average_accuracy'], 
        reverse=True
    )
    
    if sorted_results:
        best_strategy, best_stats = sorted_results[0]
        alt_acc = best_stats['average_accuracy']
        
        # 计算改进
        improvement = alt_acc - original_acc
        improvement_pct = (improvement / max(original_acc, 0.0001)) * 100 if original_acc > 0 else float('inf')
        
        color = ""
        if alt_acc > original_acc * 1.1:  # 提升超过10%
            color = "🟢"
        elif alt_acc > original_acc:
            color = "🟡"
        else:
            color = "🔴"
        
        print(f"{model_key:<45} {best_strategy:<25} {alt_acc:<10.4f} {original_acc:<12.4f} {color}")
        
        summary.append({
            'model': model_key,
            'best_alt_strategy': best_strategy,
            'alt_accuracy': alt_acc,
            'original_accuracy': original_acc,
            'improvement': improvement,
            'improvement_pct': improvement_pct
        })

print()
print("="*80)
print("📈 显著改进的模型（准确率提升超过10%）")
print("="*80)
print()

significant_improvements = [s for s in summary if s['improvement_pct'] > 10 and s['alt_accuracy'] > s['original_accuracy']]

if significant_improvements:
    print(f"{'模型':<45} {'推荐策略':<25} {'提升':<15}")
    print("-"*80)
    for item in sorted(significant_improvements, key=lambda x: x['improvement_pct'], reverse=True):
        print(f"{item['model']:<45} {item['best_alt_strategy']:<25} +{item['improvement']:.4f} ({item['improvement_pct']:.1f}%)")
else:
    print("没有模型通过替代策略获得显著提升")

print()
print("="*80)
print("📋 策略使用统计")
print("="*80)
print()

strategy_count = {}
for item in summary:
    strategy = item['best_alt_strategy']
    strategy_count[strategy] = strategy_count.get(strategy, 0) + 1

for strategy, count in sorted(strategy_count.items(), key=lambda x: x[1], reverse=True):
    print(f"  {strategy:<30}: {count:>3} 个模型")

print()
print("="*80)
print("💡 配置建议")
print("="*80)
print()

# 根据策略给出具体建议
strategy_configs = {
    '0-1000_normalized': 'export Nuoyin_API_BASE="http://your-api-base"',
    '0-100_percentage': 'export USE_PERCENTAGE_COORDS=true',
    'smart_detect': 'export NORMALIZE_PIXEL_COORDS=true',
    'direct_pixel': '# 模型输出像素坐标，可能需要修改prompt',
}

for item in significant_improvements:
    model = item['model'].split('/')[0]
    strategy = item['best_alt_strategy']
    config = strategy_configs.get(strategy, '# 无特殊配置')
    print(f"\n{model}:")
    print(f"  推荐策略: {strategy}")
    print(f"  准确率提升: {item['improvement']:.4f} ({item['improvement_pct']:.1f}%)")
    print(f"  配置: {config}")





