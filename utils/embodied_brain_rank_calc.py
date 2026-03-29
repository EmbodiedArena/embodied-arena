#!/usr/bin/env python3
"""计算 EmbodiedBrain-7B 的整体排名与具身模型内排名。"""

# 表格数据：7 列 = ObjectPerception, SpatialPerception, EmbodiedKnowledge, EmbodiedReasoning, TemporalPerception, EmbodiedNavigation, EmbodiedTaskPlanning
DATA = {
    "Cambrian-S-7B":        [58.32, 52.76, 39.77, 39.33, 39.74, 44.16, 28.33],
    "internvl3.5-8B":       [60.86, 47.65, 42.10, 41.47, 44.23, 43.40, 27.17],
    "Qwen3-vl-8B-Ins":      [64.49, 53.16, 57.82, 46.50, 52.44, 46.61, 36.33],
    "PelicanVL-7B":         [57.57, 43.68, 55.60, 44.61, 42.97, 49.40, 38.53],
    "Mimo-Embodied-7B":     [50.92, 43.99, 53.17, 43.05, 44.57, 42.98, 42.93],
    "EmbodiedBrain-7B":     [45.66, 30.20, 47.90, 41.06, 41.30, 38.07, 34.50],
    "Gemini-2.5-Pro":       [60.27, 44.17, 65.80, 52.03, 60.79, 46.28, 35.67],
    "Qwen-VL-Max":          [49.79, 38.28, 69.34, 48.55, 59.28, 41.97, 58.50],
    "RoboBrain2.0-7B":      [34.49, 31.13, 50.98, 29.51, 30.47, 35.81, 52.83],
    "Cosmos-Reason1":       [31.38, 29.07, 54.36, 32.41, 32.56, 38.70, 39.25],
    "o3":                   [60.33, 51.44, 63.38, 53.00, 60.67, 51.33, 36.08],
    "GPT-5.2":              [57.61, 44.28, 57.44, 43.30, 55.49, 49.83, 40.54],
}

ABILITIES = [
    "ObjectPerception", "SpatialPerception", "EmbodiedKnowledge",
    "EmbodiedReasoning", "TemporalPerception", "EmbodiedNavigation",
    "EmbodiedTaskPlanning",
]

EMBODIED = [
    "Cambrian-S-7B", "PelicanVL-7B", "Mimo-Embodied-7B",
    "EmbodiedBrain-7B", "RoboBrain2.0-7B", "Cosmos-Reason1",
]

def main():
    import json

    # 计算每模型 7 项平均分
    avg = {name: sum(scores) / 7 for name, scores in DATA.items()}

    # 每个能力项：整体排名 + 具身模型内排名
    per_ability_overall = []
    per_ability_embodied = []
    for idx, ability in enumerate(ABILITIES):
        # 按该能力项得分从高到低排序（整体）
        overall_sorted = sorted(DATA.keys(), key=lambda x: -DATA[x][idx])
        rank_overall = overall_sorted.index("EmbodiedBrain-7B") + 1
        # 按该能力项得分从高到低排序（具身模型内）
        embodied_sorted = sorted(EMBODIED, key=lambda x: -DATA[x][idx])
        rank_embodied = embodied_sorted.index("EmbodiedBrain-7B") + 1
        score = DATA["EmbodiedBrain-7B"][idx]
        per_ability_overall.append({
            "ability": ability,
            "score": round(score, 2),
            "rank_overall": rank_overall,
            "rank_embodied": rank_embodied,
        })

    # 整体排名（12 个模型，按平均分从高到低）
    overall = sorted(DATA.keys(), key=lambda x: -avg[x])
    overall_rank = overall.index("EmbodiedBrain-7B") + 1
    embodied_sorted = sorted(EMBODIED, key=lambda x: -avg[x])
    embodied_rank = embodied_sorted.index("EmbodiedBrain-7B") + 1
    eb_avg = avg["EmbodiedBrain-7B"]

    # ----- 打印：每个能力项排名 -----
    print("=" * 70)
    print("EmbodiedBrain-7B 各能力项排名")
    print("=" * 70)
    print(f"{'能力项':<22} {'得分':>8} {'整体排名':>10} {'具身模型内':>12}")
    print("-" * 70)
    for item in per_ability_overall:
        print(f"{item['ability']:<22} {item['score']:>8.2f} #{item['rank_overall']}/12{'':>4} #{item['rank_embodied']}/6")
    print("-" * 70)
    print(f"{'7项平均':<22} {eb_avg:>8.2f} #{overall_rank}/12{'':>4} #{embodied_rank}/6")
    print("=" * 70)

    print("\n【整体排名】(共 12 个模型，按 7 项平均分排序)")
    for i, name in enumerate(overall, 1):
        mark = " <-- EmbodiedBrain-7B" if name == "EmbodiedBrain-7B" else ""
        print(f"  #{i}  {name}: {avg[name]:.2f}{mark}")
    print(f"\n  => EmbodiedBrain-7B 整体排名: 第 {overall_rank} 名 / 12")

    print("\n【具身模型内排名】(共 6 个具身模型)")
    for i, name in enumerate(embodied_sorted, 1):
        mark = " <-- EmbodiedBrain-7B" if name == "EmbodiedBrain-7B" else ""
        print(f"  #{i}  {name}: {avg[name]:.2f}{mark}")
    print(f"\n  => EmbodiedBrain-7B 具身模型内排名: 第 {embodied_rank} 名 / 6")
    print("=" * 70)

    # 输出 JSON
    out = {
        "model": "EmbodiedBrain-7B",
        "avg_score_7": round(eb_avg, 2),
        "overall_rank": overall_rank,
        "overall_total": 12,
        "embodied_rank": embodied_rank,
        "embodied_total": 6,
        "per_ability": per_ability_overall,
    }
    with open("/your/path/to/embodied-eval-main/embodied_brain_rank_result.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("\n已写入 embodied_brain_rank_result.json")

if __name__ == "__main__":
    main()
