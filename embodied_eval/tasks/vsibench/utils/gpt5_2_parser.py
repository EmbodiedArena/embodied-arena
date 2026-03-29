import json
import re
import os
import numpy as np
import pandas as pd
from loguru import logger as eval_logger

# --- 配置路径 ---
INPUT_FILE = "/your/path/to/embodied-eval-main/logs/vsibench/gpt5_2_res/res1.txt"
OUTPUT_FILE = "/your/path/to/embodied-eval-main/logs/vsibench/gpt5_2_res/new_parser.txt"

MCA_QUESTION_TYPES = [
    "object_rel_direction_easy", "object_rel_direction_medium", "object_rel_direction_hard",
    "object_rel_distance", "route_planning", "obj_appearance_order",
]
NA_QUESTION_TYPES = [
    "object_abs_distance", "object_counting", "object_size_estimation", "room_size_estimation",
]

# --- 核心提取与计算逻辑 ---

def extract_na_number(res_str):
    """
    严格的数字提取逻辑：
    1. 使用 [0-9] 匹配，天然忽略所有 Unicode 数字符号（如 ²、³、½、≈）。
    2. 只有当 res 中仅存在 1 个独立的 ASCII 数字块时才视为有效。
    """
    if res_str is None:
        return None
    
    res_str = str(res_str)
    
    # [0-9] 确保不匹配任何非 ASCII 的数字符号
    # 匹配整数或带小数点的浮点数
    matches = re.findall(r'[0-9]+\.?[0-9]*', res_str)
    
    # 进一步过滤掉仅包含点号的非法匹配项
    nums = [n for n in matches if any(c.isdigit() for c in n)]
    
    # 严格判定：只能有且仅有 1 个数字
    if len(nums) == 1:
        try:
            return float(nums[0])
        except ValueError:
            return None
    
    # 0个数字（无结果）或多个数字（如范围 18-22 或 14m2）均返回 None
    return None

def mean_relative_accuracy(pred, target, start=.5, end=.95, interval=.05):
    """NA 类题型的相对精度计算"""
    if pred is None:
        return 0.0
    
    # 相对误差计算
    def abs_dist_norm(p, t):
        if t == 0: return 1.0 if p == 0 else 0.0
        return abs(p - t) / t

    rel_err = abs_dist_norm(pred, target)
    num_pts = (end - start) / interval + 2
    conf_intervs = np.linspace(start, end, int(num_pts))
    
    # 判断误差是否在允许范围内
    accuracy = rel_err <= (1 - conf_intervs)
    return accuracy.mean()

def calculate_mca_score(res, target):
    """MCA 题型处理：模糊匹配第一个单词"""
    if not res: return 0.0
    pred = str(res).split(" ")[0].rstrip(".").strip().lower()
    return 1.0 if pred == str(target).lower() else 0.0

# --- 汇总逻辑 ---

def aggregate_all(results_list):
    if not results_list:
        return {"error": "No data processed"}

    df = pd.DataFrame(results_list)
    output = {}

    # 1. 计算每个子类别的平均分
    for q_type, group in df.groupby("question_type"):
        metric_name = "accuracy" if q_type in MCA_QUESTION_TYPES else "MRA:.5:.95:.05"
        output[f"{q_type}_{metric_name}"] = group["score"].mean()

    # 2. 聚合方向类指标 (Easy, Medium, Hard 平均)
    dir_keys = [f"object_rel_direction_{l}_accuracy" for l in ["easy", "medium", "hard"]]
    dir_vals = [output.pop(k) for k in dir_keys if k in output]
    if dir_vals:
        output["object_rel_direction_accuracy"] = sum(dir_vals) / 3.0

    # 3. 计算大类平均分
    mca_metrics = [v for k, v in output.items() if "accuracy" in k]
    na_metrics = [v for k, v in output.items() if "MRA" in k]
    
    if mca_metrics: output["accuracy_average"] = sum(mca_metrics) / len(mca_metrics)
    if na_metrics: output["MRA:.5:.95:.05_average"] = sum(na_metrics) / len(na_metrics)

    # 4. 计算最终 Overall 分数
    all_final_vals = [v for v in output.values() if isinstance(v, (float, int))]
    output["overall"] = sum(all_final_vals) / len(all_final_vals) if all_final_vals else 0
    
    return output

# --- 主程序 ---

def main():
    processed_items = []
    
    if not os.path.exists(INPUT_FILE):
        eval_logger.error(f"找不到输入文件: {INPUT_FILE}")
        return

    eval_logger.info(f"正在处理: {INPUT_FILE}")

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            
            try:
                data = json.loads(line)
                q_type = data.get("question_type")
                res = data.get("res", "")
                target = data.get("target", "")

                pred_val = res
                
                score = 0.0
                if q_type in MCA_QUESTION_TYPES:
                    score = calculate_mca_score(res, target)
                elif q_type in NA_QUESTION_TYPES:
                    pred_val = extract_na_number(res)
                    # with open("/your/path/to/embodied-eval-main/logs/vsibench/gpt5_2_res/new_parser.txt","a",encoding="utf-8") as f:
                    #     f.write(res + " -------> " + str(pred_val) + "\n")
                    try:
                        target_val = float(target)
                    except:
                        target_val = 0.0
                    score = mean_relative_accuracy(pred_val, target_val)
                
                processed_items.append({"question_type": q_type, "score": score})
                # processed_items.append({"res": pred_val, "target": target, "question_type": q_type, "score": score})
            except Exception as e:
                eval_logger.warning(f"解析行失败: {e}")

    # 汇总结果
    final_results = aggregate_all(processed_items)

    # with open(OUTPUT_FILE, "w", encoding="utf-8") as f_out:
    #     json.dump(processed_items, f_out, indent=4, ensure_ascii=False)


    # 写入文件
    try:
        os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
        with open(OUTPUT_FILE, "a", encoding="utf-8") as f_out:
            json.dump(final_results, f_out, indent=4, ensure_ascii=False)
        eval_logger.info(f"重新计算完成！结果已存入: {OUTPUT_FILE}")
    except Exception as e:
        eval_logger.error(f"保存结果失败: {e}")

if __name__ == "__main__":
    main()