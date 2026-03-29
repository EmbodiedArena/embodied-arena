#!/usr/bin/env python3
"""
修正ERQA评估结果中的准确率问题
重新计算所有样本的accuracy并更新results文件
"""

import json
import os
import re
from pathlib import Path
from collections import defaultdict

def extract_single_word_option(text):
    """
    从预测文本中提取单个单词选项（A-D）
    处理各种响应格式，包括thinking模式输出
    """
    if not text:
        return ""
    
    # 清理文本
    text = text.strip()
    
    # 首先处理thinking模式：提取</think>标签后的内容
    think_pattern = r'</think>\s*(.+?)$'
    think_match = re.search(think_pattern, text, re.DOTALL | re.IGNORECASE)
    
    if think_match:
        # 提取</think>标签后的文本
        answer_text = think_match.group(1).strip()
        
        # 如果</think>后的答案是单个字母（A-D），直接返回
        single_letter_match = re.match(r'^([A-D])\.?\s*$', answer_text, re.IGNORECASE)
        if single_letter_match:
            return single_letter_match.group(1).upper()
        
        # 否则继续在提取的文本上进行模式匹配
        text = answer_text
    
    # 常见的答案提取模式
    answer_patterns = [
        r'(?:answer|Answer)(?:\s*is)?\s*:\s*([A-D])',  # Answer: A 或 Answer is A
        r'(?:the|The)\s+answer\s+is\s+([A-D])',  # The answer is A
        r'^\s*([A-D])\s*[\.\):]',  # 开头的单个字母带标点
        r'\b([A-D])\s*[\.\):]?\s*$',  # 结尾的单个字母
        r'^([A-D])$',  # 只有一个字母
    ]
    
    for pattern in answer_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).upper()
    
    # 如果没有模式匹配，尝试找到任何单个大写字母A-D
    fallback_match = re.search(r'\b([A-D])\b', text)
    if fallback_match:
        return fallback_match.group(1).upper()
    
    # 如果找不到有效答案，返回空字符串
    return ""

def exact_match(response_text, answer):
    """精确匹配评估"""
    return response_text.replace(".", "").strip().lower() == answer.strip().lower()

def fix_sample_file(sample_file_path):
    """修正单个样本文件的accuracy"""
    print(f"\n处理: {sample_file_path}")
    
    with open(sample_file_path, 'r', encoding='utf-8') as f:
        samples = [json.loads(line) for line in f]
    
    fixed_count = 0
    total_correct = 0
    
    for sample in samples:
        target = sample['target']
        
        if not sample['resps'] or not sample['resps'][0]:
            # 空响应，accuracy应该为False
            old_acc = sample.get('accuracy')
            sample['accuracy'] = False
            if old_acc != False:
                fixed_count += 1
            continue
        
        raw_resp = sample['resps'][0][0]
        
        # 重新提取答案
        extracted = extract_single_word_option(raw_resp)
        
        # 重新计算accuracy
        new_accuracy = exact_match(extracted, target)
        old_accuracy = sample.get('accuracy')
        
        if new_accuracy != old_accuracy:
            fixed_count += 1
            print(f"  修正 Doc ID {sample['doc_id']}: {old_accuracy} -> {new_accuracy} (提取='{extracted}', 目标='{target}')")
        
        sample['accuracy'] = new_accuracy
        
        if new_accuracy:
            total_correct += 1
    
    # 保存修正后的样本文件
    with open(sample_file_path, 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    print(f"  修正数量: {fixed_count}")
    print(f"  总正确数: {total_correct}/{len(samples)} ({total_correct/len(samples)*100:.2f}%)")
    
    return samples, fixed_count

def regenerate_results(sample_file_path, results_file_path):
    """重新生成results文件"""
    print(f"\n重新生成结果文件: {results_file_path}")
    
    with open(sample_file_path, 'r', encoding='utf-8') as f:
        samples = [json.loads(line) for line in f]
    
    # 按问题类型聚合结果
    from collections import defaultdict
    import pandas as pd
    
    ERQA_QUESTION_TYPES = [
        "Action Reasoning",
        "Multi-view Reasoning",
        "Other",
        "Pointing",
        "State Estimation",
        "Spatial Reasoning",
        "Trajectory Reasoning",
        "Task Reasoning",
    ]
    
    results = []
    for sample in samples:
        results.append({
            'question_type': sample.get('question_type', 'erqa'),
            'accuracy': sample.get('accuracy', False)
        })
    
    results_df = pd.DataFrame(results)
    output = {}
    
    for question_type, question_type_indexes in results_df.groupby("question_type").groups.items():
        per_question_type = results_df.iloc[question_type_indexes]
        if question_type in ERQA_QUESTION_TYPES:
            output[f"{question_type}_accuracy"] = float(per_question_type['accuracy'].mean())
    
    # 计算平均值
    metric_to_values = defaultdict(list)
    for key, val in output.items():
        if "_" in key:
            qtype, metric_name = key.rsplit("_", 1)
            if isinstance(val, (float, int)):
                metric_to_values[metric_name].append(val)
    
    for metric_name, vals in metric_to_values.items():
        if len(vals) > 0:
            avg_val = sum(vals) / len(vals)
            output[f"{metric_name}_average"] = avg_val
    
    output["overall"] = sum([_ for _ in output.values()]) / len(output)
    
    # 保存结果文件
    with open(results_file_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    print(f"  Overall Accuracy: {output['overall']:.4f}")
    
    return output

def main():
    logs_dir = Path('/your/path/to/embodied-eval-main/logs/erqa')
    
    total_fixed = 0
    processed_models = []
    
    # 遍历所有模型的结果
    for model_dir in sorted(logs_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        
        print(f"\n{'='*80}")
        print(f"处理模型: {model_dir.name}")
        print('='*80)
        
        # 查找该模型下的所有样本文件
        sample_files = list(model_dir.rglob('samples_erqa.json'))
        
        for sample_file in sample_files:
            try:
                # 修正样本文件
                samples, fixed_count = fix_sample_file(sample_file)
                total_fixed += fixed_count
                
                # 重新生成结果文件
                results_file = sample_file.parent / 'results_erqa.json'
                if results_file.exists():
                    output = regenerate_results(sample_file, results_file)
                    processed_models.append({
                        'model': model_dir.name,
                        'timestamp': sample_file.parent.name,
                        'fixed_count': fixed_count,
                        'overall_accuracy': output['overall']
                    })
                
            except Exception as e:
                print(f"  ❌ 错误: {e}")
                import traceback
                traceback.print_exc()
    
    # 总结
    print(f"\n{'='*80}")
    print("修正完成总结")
    print('='*80)
    print(f"总修正数量: {total_fixed}")
    print(f"处理模型数: {len(processed_models)}")
    
    # 保存修正摘要
    summary_file = logs_dir / 'accuracy_fix_summary.json'
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump({
            'total_fixed': total_fixed,
            'models': processed_models
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\n修正摘要已保存到: {summary_file}")

if __name__ == '__main__':
    main()

