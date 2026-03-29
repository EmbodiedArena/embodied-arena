#!/usr/bin/env python3
"""
分析ERQA评估结果的准确性
检查是否存在答案提取错误或评估不准确的情况
"""

import json
import os
import sys
from pathlib import Path
from collections import defaultdict

# 从process.py导入提取函数
sys.path.insert(0, os.path.dirname(__file__))
from process import extract_single_word_option, exact_match

def analyze_samples(sample_file):
    """分析单个样本文件"""
    issues = []
    
    with open(sample_file, 'r', encoding='utf-8') as f:
        samples = [json.loads(line) for line in f]
    
    for sample in samples:
        doc_id = sample['doc_id']
        target = sample['target']
        
        if not sample['resps'] or not sample['resps'][0]:
            issues.append({
                'doc_id': doc_id,
                'issue': 'empty_response',
                'target': target
            })
            continue
        
        raw_resp = sample['resps'][0][0]
        
        # 重新提取答案
        extracted = extract_single_word_option(raw_resp)
        
        # 检查是否与样本中的accuracy字段一致
        expected_acc = exact_match(extracted, target)
        actual_acc = sample.get('accuracy')
        
        # 检查问题类型
        if expected_acc != actual_acc:
            issues.append({
                'doc_id': doc_id,
                'issue': 'accuracy_mismatch',
                'raw_response': raw_resp,
                'extracted': extracted,
                'target': target,
                'expected_accuracy': expected_acc,
                'actual_accuracy': actual_acc
            })
        
        # 检查提取失败的情况
        if not extracted and raw_resp:
            issues.append({
                'doc_id': doc_id,
                'issue': 'extraction_failed',
                'raw_response': raw_resp[:200],
                'target': target
            })
        
        # 检查可能的误判：响应中包含正确答案但提取失败
        if not expected_acc and target.upper() in raw_resp.upper():
            issues.append({
                'doc_id': doc_id,
                'issue': 'potential_false_negative',
                'raw_response': raw_resp[:200],
                'extracted': extracted,
                'target': target
            })
    
    return issues, samples

def main():
    logs_dir = Path('/your/path/to/embodied-eval-main/logs/erqa')
    
    all_issues = defaultdict(list)
    
    # 遍历所有模型的结果
    for model_dir in logs_dir.iterdir():
        if not model_dir.is_dir():
            continue
        
        print(f"\n{'='*80}")
        print(f"分析模型: {model_dir.name}")
        print('='*80)
        
        # 查找该模型下的所有样本文件
        sample_files = list(model_dir.rglob('samples_erqa.json'))
        
        for sample_file in sample_files:
            print(f"\n检查文件: {sample_file.relative_to(logs_dir)}")
            
            try:
                issues, samples = analyze_samples(sample_file)
                
                print(f"总样本数: {len(samples)}")
                print(f"发现问题数: {len(issues)}")
                
                # 按问题类型分组
                issue_types = defaultdict(list)
                for issue in issues:
                    issue_types[issue['issue']].append(issue)
                
                for issue_type, type_issues in issue_types.items():
                    print(f"\n  {issue_type}: {len(type_issues)} 个")
                    all_issues[issue_type].extend(type_issues)
                    
                    # 显示前3个例子
                    for i, issue in enumerate(type_issues[:3]):
                        print(f"    例子 {i+1}:")
                        if 'raw_response' in issue:
                            print(f"      原始响应: {issue['raw_response'][:100]}")
                        if 'extracted' in issue:
                            print(f"      提取结果: {issue['extracted']}")
                        if 'target' in issue:
                            print(f"      目标答案: {issue['target']}")
                        if 'expected_accuracy' in issue:
                            print(f"      预期准确: {issue['expected_accuracy']}")
                            print(f"      实际准确: {issue['actual_accuracy']}")
                
            except Exception as e:
                print(f"  错误: {e}")
                import traceback
                traceback.print_exc()
    
    # 总结
    print(f"\n{'='*80}")
    print("总体问题汇总")
    print('='*80)
    for issue_type, issues in all_issues.items():
        print(f"{issue_type}: {len(issues)} 个")
    
    # 保存详细报告
    report_file = '/your/path/to/embodied-eval-main/embodied_eval/tasks/erqa/accuracy_analysis_report.json'
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(dict(all_issues), f, ensure_ascii=False, indent=2)
    
    print(f"\n详细报告已保存到: {report_file}")

if __name__ == '__main__':
    main()



