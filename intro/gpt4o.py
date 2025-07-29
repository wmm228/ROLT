import json

input_path = "gpt4o_scored.json"
output_path = "position_wise_avg_scores.txt"
num_positions = 10

# 初始化每个位置的得分列表
position_scores = [[] for _ in range(num_positions)]

# 读取数据
with open(input_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# 收集每个子问题位置上的所有分数
for item in data:
    subquestions = item.get("subquestions", [])
    for i in range(min(len(subquestions), num_positions)):
        score = subquestions[i].get("score")
        if isinstance(score, (int, float)):
            position_scores[i].append(score)

# 计算平均值并写入文件
with open(output_path, "w", encoding="utf-8") as out:
    for i, scores in enumerate(position_scores):
        if scores:
            avg = sum(scores) / len(scores)
            out.write(f"Position {i+1}: {avg:.4f}\n")
            print(f"Position {i+1}: 平均相似度 = {avg:.4f}")
        else:
            out.write(f"Position {i+1}: 无有效分数\n")
            print(f"Position {i+1}: 无有效分数")
