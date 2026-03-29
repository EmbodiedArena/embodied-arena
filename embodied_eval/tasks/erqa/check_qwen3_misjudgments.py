#!/usr/bin/env python3
"""
检查Qwen3-VL-8B-Instruct失败案例中是否存在误判
"""

import json
import re
import sys
sys.path.insert(0, '/your/path/to/embodied-eval-main/embodied_eval/tasks/erqa')
from process import extract_single_word_option, exact_match

def check_for_misjudgments():
    sample_file = '/your/path/to/embodied-eval-main/logs/erqa/qwen3-vl-8b-instruct/20260121_211254/samples_erqa.json'
    
    with open(sample_file, 'r') as f:
        samples = [json.loads(line) for line in f]
    
    print("="*80)
    print("检查 Qwen3-VL-8B-Instruct 失败案例中的潜在误判")
    print("="*80)
    
    # 只看失败的案例
    wrong_samples = [s for s in samples if not s.get('accuracy', False)]
    
    potential_issues = []
    
    print(f"\n总失败案例数: {len(wrong_samples)}")
    print("\n正在检查每个失败案例...\n")
    
    for i, sample in enumerate(wrong_samples, 1):
        doc_id = sample['doc_id']
        target = sample['target']
        question = sample['doc']
        
        if not sample['resps'] or not sample['resps'][0]:
            continue
        
        raw_response = sample['resps'][0][0]
        
        # 重新提取答案
        extracted = extract_single_word_option(raw_response)
        
        # 检查是否应该是正确的
        should_be_correct = exact_match(extracted, target)
        
        if should_be_correct:
            potential_issues.append({
                'doc_id': doc_id,
                'issue_type': 'accuracy_field_wrong',
                'question': question[:150],
                'raw_response': raw_response,
                'extracted': extracted,
                'target': target,
                'question_type': sample.get('question_type', 'Unknown')
            })
            print(f"⚠️ 发现潜在误判 - Doc ID {doc_id}")
            print(f"  提取答案: '{extracted}' == 目标答案: '{target}' (应该正确)")
        
        # 检查是否答案在响应中但提取失败
        if not extracted:
            # 检查目标答案是否在原始响应中
            if target.upper() in raw_response.upper():
                potential_issues.append({
                    'doc_id': doc_id,
                    'issue_type': 'extraction_may_fail',
                    'question': question[:150],
                    'raw_response': raw_response[:200],
                    'extracted': extracted,
                    'target': target,
                    'question_type': sample.get('question_type', 'Unknown')
                })
                print(f"🔍 提取可能有问题 - Doc ID {doc_id}")
                print(f"  目标答案 '{target}' 在响应中但未被提取")
        
        # 检查响应是否包含多个选项字母（可能是详细解释）
        option_letters = re.findall(r'\b([A-D])\b', raw_response)
        if len(option_letters) > 1 and extracted != target:
            # 检查目标答案是否在其中
            if target in option_letters:
                potential_issues.append({
                    'doc_id': doc_id,
                    'issue_type': 'multiple_options_in_response',
                    'question': question[:150],
                    'raw_response': raw_response[:300],
                    'extracted': extracted,
                    'target': target,
                    'all_options_found': option_letters,
                    'question_type': sample.get('question_type', 'Unknown')
                })
                print(f"📝 响应包含多个选项 - Doc ID {doc_id}")
                print(f"  找到的选项: {option_letters}, 提取: '{extracted}', 目标: '{target}'")
    
    # 详细报告
    print("\n" + "="*80)
    print("详细分析结果")
    print("="*80)
    
    issue_types = {}
    for issue in potential_issues:
        issue_type = issue['issue_type']
        if issue_type not in issue_types:
            issue_types[issue_type] = []
        issue_types[issue_type].append(issue)
    
    for issue_type, issues in issue_types.items():
        print(f"\n{'='*80}")
        print(f"问题类型: {issue_type}")
        print(f"数量: {len(issues)}")
        print('='*80)
        
        for i, issue in enumerate(issues[:10], 1):  # 只显示前10个
            print(f"\n案例 {i} - Doc ID: {issue['doc_id']} ({issue['question_type']})")
            print(f"问题: {issue['question']}...")
            print(f"原始响应: {issue['raw_response'][:200]}...")
            print(f"提取答案: '{issue['extracted']}'")
            print(f"目标答案: '{issue['target']}'")
            if 'all_options_found' in issue:
                print(f"响应中找到的所有选项: {issue['all_options_found']}")
    
    # 保存报告
    report = {
        'model': 'qwen3-vl-8b-instruct',
        'total_wrong_samples': len(wrong_samples),
        'potential_issues_count': len(potential_issues),
        'issues_by_type': {
            issue_type: len(issues)
            for issue_type, issues in issue_types.items()
        },
        'detailed_issues': potential_issues
    }
    
    report_file = '/your/path/to/embodied-eval-main/logs/erqa/qwen3-vl-8b-instruct/misjudgment_check_report.json'
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"\n\n{'='*80}")
    print("总结")
    print('='*80)
    print(f"总失败案例数: {len(wrong_samples)}")
    print(f"潜在问题数: {len(potential_issues)}")
    
    if len(potential_issues) == 0:
        print("\n✅ 未发现明显的误判情况，所有失败案例的判定看起来都是正确的。")
    else:
        print(f"\n⚠️ 发现 {len(potential_issues)} 个潜在问题，需要进一步检查。")
        print("\n问题分类:")
        for issue_type, count in report['issues_by_type'].items():
            print(f"  - {issue_type}: {count} 个")
    
    print(f"\n💾 详细报告已保存到: {report_file}")

if __name__ == '__main__':
    check_for_misjudgments()



