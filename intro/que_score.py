import os
import json
import torch
import traceback
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
import openai

# === 限定使用 GPU 5,6,7 ===
os.environ["CUDA_VISIBLE_DEVICES"] = "5,6,7"
device = "cuda" if torch.cuda.is_available() else "cpu"

# === OpenAI 设置 ===
openai.api_key = "sk-v3AJ9FgF4gRzR4sGCC622rsA9AOQCIaQIqjF2QRdTsACN4S2"
openai.api_base = "https://api.agicto.cn/v1"

# === 加载 SentenceTransformer BGE 模型 ===
print("[INFO] 加载 BGE-M3 模型（SentenceTransformer方式）...")
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(BASE_DIR, 'bge_m3')
bge_model = SentenceTransformer(model_path, device=device)

# === 加载前 num 条问题 ===
def load_questions(path, num=100):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return [item['question'] for item in data[:num]]

# === GPT 子问题生成函数 ===
def gpt_decompose(question, B=10):
    history_prompt = f"Original Question: {question}"
    prompt = (
        f"You are an expert in multi-hop question decomposition. Let's think step by step.\n"
        f"Given the following reasoning history, generate exactly {B} new sub-questions that logically extend this reasoning to help answer the original question.\n\n"
        f"History:\n{history_prompt}\n\n"
        f"Each sub-question must be a relevant, self-contained atomic question—clear, specific, answerable, and not redundant.\n"
        f"Each sub-question should focus on resolving the next step needed to answer the original question, potentially leveraging key terms from the original question and reasoning history.\n\n"
        f"Output exactly {B} sub-questions in the following format. No explanations or commentary:\n"
        + "\n".join([f"{i+1}." for i in range(B)])
    )

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )
    content = response["choices"][0]["message"]["content"]
    lines = [line.strip() for line in content.strip().split("\n") if line.strip()]
    subquestions = []
    for line in lines:
        if '.' in line:
            sub = line.split('.', 1)[1].strip()
            if sub:
                subquestions.append(sub)
    return subquestions[:B]

# === 相似度打分函数（sentence-transformers + util.cos_sim）===
def compute_similarity_bge(main_question, subquestions):
    embeddings = bge_model.encode([main_question] + subquestions, convert_to_tensor=True)
    main_emb = embeddings[0]
    sub_embs = embeddings[1:]
    scores = util.cos_sim(main_emb, sub_embs)[0].tolist()
    return scores

# === 主流程 ===
def main():
    input_path = "hotpot_dev_fullwiki_v1.json"
    output_path = "gpt3.5.json"
    num_questions = 100
    B = 10

    questions = load_questions(input_path, num=num_questions)
    all_results = []
    avg_score_file = open("avg_similarity_scores.txt", "w", encoding="utf-8")

    for idx, q in enumerate(tqdm(questions, desc="Processing Questions", total=len(questions))):
        try:
            sub_qs = gpt_decompose(q, B=B)

            if len(sub_qs) < B:
                print(f"[Skip] Q{idx+1}: GPT 子问题不足 -> {len(sub_qs)}")
                continue

            scores = compute_similarity_bge(q, sub_qs)

            if len(scores) != B:
                print(f"[Skip] Q{idx+1}: 相似度分数数量不符 -> {len(scores)}")
                continue

            avg_score = sum(scores) / len(scores)
            print(f"[Avg] Q{idx+1}: 平均相似度得分 = {avg_score:.4f}")
            avg_score_file.write(f"Q{idx+1}: {avg_score:.4f}\n")

            result = {
                "question": q,
                "subquestions": [
                    {"text": sub_qs[i], "score": scores[i]} for i in range(B)
                ]
            }
            all_results.append(result)

        except Exception as e:
            print(f"[Error] Q{idx+1}: {q}\nException: {repr(e)}")
            traceback.print_exc()
            continue

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    avg_score_file.close()
    print(f"\n✅ 已保存 {len(all_results)} 条记录到 {output_path}")
    print(f"✅ 平均得分已写入 avg_similarity_scores.txt")


if __name__ == "__main__":
    main()
