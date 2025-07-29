import json
import pandas as pd
from simpletransformers.classification import ClassificationModel
import os
import torch
from llm_client import LLMClient
from mcts_all import *
import re
import string
from bge import BGEEmbedder
import sys

def normalize_answer(s):
    """标准化答案，去掉标点、大小写、空格"""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        return ''.join(ch for ch in text if ch not in set(string.punctuation))

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

# ------------------ 配置 ------------------
BEST_MODEL_PATH = "./bert_base_uncased_output/best_model"
os.environ["CUDA_VISIBLE_DEVICES"] = "5,6,7"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ClassificationModel(
    "bert", BEST_MODEL_PATH, num_labels=2,
    use_cuda=True, args={"device": "cuda"}
)

# ------------------ 参数映射、动态设置 ------------------
def map_hop_label(label: int) -> int:
    return {0: 2, 1: 3}.get(label, 2)

def determine_max_iter(hop: int) -> int:
    return {2: 5, 3: 15, 4: 20}.get(hop, 15) # 7，12，20 迭代次数

# ------------------ 数据读取 ------------------
def load_hotpotqa_data(file_path: str) -> list:
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return [(item['question'], process_context(item['context']), item['answer']) for item in data]

def process_context(context_list: list) -> list:
    return [sentence for _, content in context_list for sentence in content]

def em_cover(final_answer, gold_answer):
    """
    只要最终答案包含标准答案就算覆盖成功。
    """
    return int(normalize_answer(gold_answer) in normalize_answer(final_answer))

def f1_score(prediction, ground_truth):
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()
    common = set(pred_tokens) & set(gt_tokens)
    if len(common) == 0:
        return 0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gt_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def exact_match_score(prediction, ground_truth):
    return int(normalize_answer(prediction) == normalize_answer(ground_truth))

def recall_score(prediction, ground_truth):
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()
    common = set(pred_tokens) & set(gt_tokens)
    if len(gt_tokens) == 0:
        return 0
    return len(common) / len(gt_tokens)

def evaluate_results(results, data):
    """
    results: [{'question': q, 'answer': final_answer}, ...]
    data: [(question, context, gold_answer), ...]
    """
    total = len(results)
    em_total, f1_total, recall_total, em_cover_total = 0, 0, 0, 0

    for res, data_item in zip(results, data):
        pred = res["answer"]  # 模型输出
        gold = data_item[2]   # 数据集标准答案 (gold_answer)

        em_total += exact_match_score(pred, gold)
        f1_total += f1_score(pred, gold)
        recall_total += recall_score(pred, gold)
        em_cover_total += em_cover(pred, gold)  # ⚠️ EMCover = final_answer 包含标准答案

    sys.stdout.flush()  # 强制刷新缓冲区，立即输出
    print(f"\n[FINAL EVALUATION]")
    print(f"总问题数: {total}")
    print(f"Exact Match (EM): {em_total / total:.4f}")
    print(f"F1 Score: {f1_total / total:.4f}")
    print(f"Recall: {recall_total / total:.4f}")
    print(f"EMCover: {em_cover_total / total:.4f}")
    sys.stdout.flush()  # 强制刷新缓冲区，立即输出

    return em_total / total, f1_total / total, recall_total / total, em_cover_total / total


# ------------------ 主函数 ------------------
def main():
    # **你的 key 和地址，直接写这里**
    base_url = "https://api.agicto.cn/v1"
    api_key = "sk-YEo77VyFtv43BlCE3xZLTFZHGygUgCbiPvOIplzEqeHKggQH"

    # 初始化 LLM Client
    llm_client = LLMClient(api_key=api_key, base_url=base_url)
    
    # 初始化 bgeembedder
    bgeembedder = BGEEmbedder(model_name = "bge_m3", embeddings_dir="embeddings_parts")

    # 配置其他参数
    dataset_path = "hotpot_dev_fullwiki_v1.json"
    B = 4 # 每步生成子问题数，剪枝数
    results = []

    # 读取数据
    data = load_hotpotqa_data(dataset_path)
    print(f"[INFO] 加载 {len(data)} 条数据")
    sys.stdout.flush()  # 强制刷新缓冲区，立即输出

    # 遍历每个问题
    for idx, (question, context, answer) in enumerate(data[:500]):  # 测试只跑前1条数据
        print(f"\n[INFO] 问题 {idx+1}: {question}")
        print(answer)
        sys.stdout.flush()  # 强制刷新缓冲区，立即输出
        
        # Step 1: 跳数预测
        hop_prediction, _ = model.predict([question])
        max_hop = map_hop_label(hop_prediction[0])
        max_iter = determine_max_iter(max_hop)
        print(f"[INFO] 跳数预测为: {max_hop} 跳, 最大迭代次数: {max_iter}")
        sys.stdout.flush()  # 强制刷新缓冲区，立即输出

        # Step 2: MCTS 搜索 (直接返回最终答案)
        root = MCTSNode(question=question, answer=None, depth=0)
        final_answer = mcts_search(
            root=root,
            max_hop=max_hop,
            max_iter=max_iter,
            llm_client=llm_client,
            question=question,
            context=context,
            B=max_hop,
            bgeembedder = bgeembedder
        )

        print(f"[RESULT] 最终答案: {final_answer}")
        sys.stdout.flush()  # 强制刷新缓冲区，立即输出

        # 收集结果
        results.append({"question": question, "answer": final_answer})

    # ✅ 评估所有指标
    evaluate_results(results, data)
    # 保存所有答案
    pd.DataFrame(results).to_csv("mcts_hotpotqa_results.csv", index=False)
    print("[INFO] 所有结果已保存至 mcts_hotpotqa_results.csv")

# ------------------ 脚本直接调用 ------------------
if __name__ == "__main__":
    main()
