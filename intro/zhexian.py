import matplotlib.pyplot as plt

# ==== 可调超参数区域 ====

# 所有输入文件及其图例标签
txt_files = {
    "gpt3.5_sim.txt": "gpt-3.5-turbo",
    "gpt4o_sim.txt": "gpt-4o-mini",
    "gpt3.5_score.txt": "gpt-3.5-turbo",
    "gpt4o_score.txt": "gpt-4o-mini"
}

# 按子图分组
sim_files = ["gpt3.5_sim.txt", "gpt4o_sim.txt"]
score_files = ["gpt3.5_score.txt", "gpt4o_score.txt"]

# 横轴范围：子问题位置 1~10
x_range = list(range(1, 11))

# 图像参数
figsize = (12, 5)                  # 整体图像尺寸 (宽, 高)
linewidth = 2                      # 折线粗细
markersize = 6                     # 标记点大小
title_y_offset = -0.15            # 标题下移比例（默认单位为轴范围）
output_file = "subquestion_scores_comparison.png"  # 输出文件名
output_dpi = 300                   # 输出分辨率

# ==== 工具函数：读取得分 ====
def read_scores(file_path):
    scores = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if ":" in line:
                try:
                    score = float(line.strip().split(":")[1])
                    scores.append(score)
                except ValueError:
                    continue
    return scores

# ==== 创建图像 ====
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

# --- 左子图：BGE 相似度 ---
for f in sim_files:
    scores = read_scores(f)
    ax1.plot(x_range, scores, label=txt_files[f], linewidth=linewidth, marker='o', markersize=markersize)

ax1.set_ylabel("Similarity Score", fontsize=12)
ax1.set_xticks(x_range)
ax1.grid(True)
ax1.legend()
ax1.set_title("(a) BGE-Based Similarity", fontsize=14, y=title_y_offset)

# --- 右子图：LLM 打分 ---
for f in score_files:
    scores = read_scores(f)
    ax2.plot(x_range, scores, label=txt_files[f], linewidth=linewidth, marker='o', markersize=markersize)

ax2.set_ylabel("LLM Score (0–100)", fontsize=12)
ax2.set_xticks(x_range)
ax2.grid(True)
ax2.legend()
ax2.set_title("(b) LLM-Based Relevance Score", fontsize=14, y=title_y_offset)

# === 布局与保存 ===
plt.tight_layout()
plt.savefig(output_file, dpi=output_dpi)
plt.show()
