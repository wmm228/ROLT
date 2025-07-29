import math
from typing import List, Dict
from collections import Counter


# ------------------ 1. 多路径答案融合 (投票) ------------------
def majority_voting(paths: List[Dict]) -> str:
    """
    从多条路径融合答案，基于简单投票策略。
    
    参数：
        paths: [{'qas': [(q, a), ...], 'reward': float}]
    
    返回：
        最终融合答案 (str)
    """
    answers = []
    for path in paths:
        qas = path['qas']
        if qas:  # 保证路径非空
            final_qa = qas[-1]  # 取最后一跳答案
            answers.append(final_qa[1].strip())

    if not answers:
        return "无法确定答案"

    # 投票选最多的答案
    counter = Counter(answers)
    final_answer, _ = counter.most_common(1)[0]
    return final_answer


# ------------------ 2. 语义相似度 (余弦相似度) ------------------
def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """
    计算两个嵌入向量的余弦相似度。
    """
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a * a for a in vec1))
    norm2 = math.sqrt(sum(b * b for b in vec2))
    return dot_product / (norm1 * norm2 + 1e-8)


# ------------------ 3. 子问题链格式化 ------------------
def format_path_for_answer(path: Dict) -> str:
    """
    将路径格式化成文本，供最终 LLM 汇总成答案。
    
    参数：
        path: {'qas': [(q, a), ...], 'reward': float}
    
    返回：
        格式化文本
    """
    qas_list = path['qas']
    formatted = []
    for idx, (q, a) in enumerate(qas_list, 1):
        formatted.append(f"[第{idx}跳]\n子问题: {q}\n答案: {a}\n")
    return "\n".join(formatted)


# ------------------ 4. 路径调试打印 (辅助) ------------------
def print_path(path: Dict):
    """
    打印路径详细信息，调试用。
    """
    print("[路径详情]")
    for idx, (q, a) in enumerate(path['qas'], 1):
        print(f"[第{idx}跳] 子问题: {q} -> 答案: {a}")
    print(f"路径得分 (reward): {path['reward']}\n")
