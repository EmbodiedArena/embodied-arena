#!/usr/bin/env python3
"""
详细分析 Qwen3-VL-8B-Instruct 在 ERQA 上的表现
"""

import json
from collections import defaultdict
import sys

def analyze_error_patterns():
    """分析错误模式"""
    sample_file = '/your/path/to/embodied-eval-main/logs/erqa/qwen3-vl-8b-instruct/20260121_211254/samples_erqa.json'
    
    with open(sample_file, 'r') as f:
        samples = [json.loads(line) for line in f]
    
    print("="*80)
    print("Qwen3-VL-8B-Instruct 在 ERQA 上的详细分析")
    print("="*80)
    
    # 基本统计
    total = len(samples)
    correct = sum(1 for s in samples if s.get('accuracy', False))
    wrong = total - correct
    
    print(f"\n📊 基本统计:")
    print(f"  总样本数: {total}")
    print(f"  正确数: {correct} ({correct/total*100:.2f}%)")
    print(f"  错误数: {wrong} ({wrong/total*100:.2f}%)")
    
    # 按问题类型分析
    print(f"\n📋 各问题类型详细表现:")
    print("-"*80)
    
    stats_by_type = defaultdict(lambda: {'correct': 0, 'wrong': 0, 'total': 0})
    
    for sample in samples:
        qtype = sample.get('question_type', 'Unknown')
        stats_by_type[qtype]['total'] += 1
        if sample.get('accuracy', False):
            stats_by_type[qtype]['correct'] += 1
        else:
            stats_by_type[qtype]['wrong'] += 1
    
    # 按准确率排序
    sorted_types = sorted(stats_by_type.items(), 
                         key=lambda x: x[1]['correct']/x[1]['total'] if x[1]['total'] > 0 else 0,
                         reverse=True)
    
    print(f"{'问题类型':<30} {'正确':<8} {'错误':<8} {'总数':<8} {'准确率':<10} {'表现'}")
    print("-"*80)
    
    for qtype, stats in sorted_types:
        acc = stats['correct'] / stats['total'] * 100 if stats['total'] > 0 else 0
        
        # 判断表现
        if acc >= 50:
            performance = "✅ 优秀"
        elif acc >= 40:
            performance = "🟡 良好"
        elif acc >= 30:
            performance = "🟠 一般"
        else:
            performance = "❌ 较差"
        
        print(f"{qtype:<30} {stats['correct']:<8} {stats['wrong']:<8} {stats['total']:<8} {acc:>6.2f}%    {performance}")
    
    # 分析错误答案的分布
    print(f"\n🔍 错误答案分析:")
    print("-"*80)
    
    wrong_samples = [s for s in samples if not s.get('accuracy', False)]
    
    # 统计预测答案和目标答案的分布
    pred_distribution = defaultdict(int)
    target_distribution = defaultdict(int)
    confusion_matrix = defaultdict(lambda: defaultdict(int))
    
    for sample in wrong_samples:
        if sample['resps'] and sample['resps'][0]:
            pred = sample['resps'][0][0].strip()
            if pred in ['A', 'B', 'C', 'D']:
                pred_distribution[pred] += 1
                target = sample['target']
                target_distribution[target] += 1
                confusion_matrix[target][pred] += 1
    
    print(f"\n错误样本中的答案分布:")
    print(f"  预测答案: A={pred_distribution['A']}, B={pred_distribution['B']}, C={pred_distribution['C']}, D={pred_distribution['D']}")
    print(f"  正确答案: A={target_distribution['A']}, B={target_distribution['B']}, C={target_distribution['C']}, D={target_distribution['D']}")
    
    # 混淆矩阵
    print(f"\n混淆矩阵 (正确答案 -> 错误预测):")
    print(f"{'实际':<10} {'预测A':<10} {'预测B':<10} {'预测C':<10} {'预测D':<10}")
    print("-"*50)
    for target in ['A', 'B', 'C', 'D']:
        print(f"{target:<10} {confusion_matrix[target]['A']:<10} {confusion_matrix[target]['B']:<10} {confusion_matrix[target]['C']:<10} {confusion_matrix[target]['D']:<10}")
    
    # 找出最难的问题类型中的典型错误
    print(f"\n❌ 最弱问题类型的典型错误案例:")
    print("="*80)
    
    # 找出准确率最低的3个类型
    weakest_types = sorted(stats_by_type.items(), 
                          key=lambda x: x[1]['correct']/x[1]['total'] if x[1]['total'] > 0 else 0)[:3]
    
    for qtype, stats in weakest_types:
        print(f"\n📌 {qtype} (准确率: {stats['correct']/stats['total']*100:.2f}%)")
        print("-"*80)
        
        # 找出该类型的错误案例
        type_wrong_samples = [s for s in wrong_samples if s.get('question_type') == qtype]
        
        for i, sample in enumerate(type_wrong_samples[:3], 1):
            print(f"\n  错误案例 {i} (Doc ID: {sample['doc_id']}):")
            print(f"    问题: {sample['doc'][:200]}...")
            resp = sample['resps'][0][0] if sample['resps'] and sample['resps'][0] else 'No response'
            print(f"    模型回答: {resp}")
            print(f"    正确答案: {sample['target']}")
    
    # 找出最强的问题类型
    print(f"\n\n✅ 最强问题类型的典型成功案例:")
    print("="*80)
    
    strongest_types = sorted(stats_by_type.items(), 
                            key=lambda x: x[1]['correct']/x[1]['total'] if x[1]['total'] > 0 else 0,
                            reverse=True)[:2]
    
    correct_samples = [s for s in samples if s.get('accuracy', False)]
    
    for qtype, stats in strongest_types:
        print(f"\n📌 {qtype} (准确率: {stats['correct']/stats['total']*100:.2f}%)")
        print("-"*80)
        
        # 找出该类型的正确案例
        type_correct_samples = [s for s in correct_samples if s.get('question_type') == qtype]
        
        for i, sample in enumerate(type_correct_samples[:2], 1):
            print(f"\n  成功案例 {i} (Doc ID: {sample['doc_id']}):")
            print(f"    问题: {sample['doc'][:200]}...")
            resp = sample['resps'][0][0] if sample['resps'] and sample['resps'][0] else 'No response'
            print(f"    模型回答: {resp}")
            print(f"    正确答案: {sample['target']}")
    
    # 保存详细分析报告
    report = {
        'model': 'qwen3-vl-8b-instruct',
        'total_samples': total,
        'correct': correct,
        'wrong': wrong,
        'overall_accuracy': correct / total,
        'performance_by_type': {
            qtype: {
                'correct': stats['correct'],
                'wrong': stats['wrong'],
                'total': stats['total'],
                'accuracy': stats['correct'] / stats['total'] if stats['total'] > 0 else 0
            }
            for qtype, stats in stats_by_type.items()
        },
        'confusion_matrix': {
            target: dict(preds) 
            for target, preds in confusion_matrix.items()
        }
    }
    
    report_file = '/your/path/to/embodied-eval-main/logs/erqa/qwen3-vl-8b-instruct/detailed_analysis_report.json'
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"\n\n💾 详细分析报告已保存到: {report_file}")

if __name__ == '__main__':
    analyze_error_patterns()



