import json
from collections import defaultdict

file_path = "/your/path/to/embodied-eval-main/logs/gpt5_2_res/res1.txt"  # 改成你的文件路径

count = defaultdict(int)
metric_sum = defaultdict(float)
metric_cnt = defaultdict(int)
metric_name = {}  # 记录每个 question_type 用的是哪种 metric

with open(file_path, "r") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue

        data = json.loads(line)
        qtype = data.get("question_type")
        if qtype is None:
            continue

        count[qtype] += 1

        if "accuracy" in data:
            val = data["accuracy"]
            metric = "accuracy"
        elif "MRA:.5:.95:.05" in data:
            val = data["MRA:.5:.95:.05"]
            metric = "MRA:.5:.95:.05"
        else:
            continue

        metric_sum[qtype] += val
        metric_cnt[qtype] += 1
        metric_name[qtype] = metric


print("question_type 统计结果")
print("=" * 70)
for qtype in count:
    avg = (
        metric_sum[qtype] / metric_cnt[qtype]
        if metric_cnt[qtype] > 0 else None
    )

    print(f"{qtype}")
    print(f"  样本数        : {count[qtype]}")
    print(f"  使用指标      : {metric_name.get(qtype)}")
    print(f"  指标均值      : {avg}")
    print("-" * 70)
