import os
import json
import openai
import traceback
from tqdm import tqdm
import time

# === OpenAI 设置 ===
openai.api_key = "sk-v3AJ9FgF4gRzR4sGCC622rsA9AOQCIaQIqjF2QRdTsACN4S2"
openai.api_base = "https://api.agicto.cn/v1"  # 如果用 openai 官方请改回 https://api.openai.com/v1

# === GPT 打分提示 ===
def gpt_score_subquestion(main_question, subquestion):
    prompt = f"""You are an expert question decomposition evaluator.

Your task is to score how relevant a given sub-question is for helping answer the original main question.

The score must be a single number between 0 and 100, where:

- 100 means the sub-question is highly relevant and essential to answering the main question.
- 0 means the sub-question is completely unrelated or unhelpful.
- Intermediate scores reflect partial relevance.

Do not provide any explanation. Just return the number.

Main Question:
{main_question}

Sub-question:
{subquestion}

Score:"""

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4.1",  # 可改为 "gpt-3.5-turbo"
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        content = response["choices"][0]["message"]["content"].strip()
        score = float(content)
        return max(0.0, min(score, 100.0))  # Clamp 0–100
    except Exception as e:
        print(f"[Error] GPT评分失败: {repr(e)}")
        return None

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
        model="gpt-4o-mini",  # 可切换为 gpt-4
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

# === 主流程 ===
def main():
    input_path = "hotpot_dev_fullwiki_v1.json"
    output_path = "gpt4o_scored.json"
    num_questions = 100
    B = 10

    with open(input_path, 'r', encoding='utf-8') as f:
        data_raw = json.load(f)

    questions = [item["question"] for item in data_raw[:num_questions]]
    all_results = []
    avg_score_file = open("avg_similarity_scores.txt", "w", encoding="utf-8")

    for idx, q in enumerate(tqdm(questions, desc="Processing Questions", total=len(questions))):
        try:
            sub_qs = gpt_decompose(q, B=B)

            if len(sub_qs) < B:
                print(f"[Skip] Q{idx+1}: GPT 子问题不足 -> {len(sub_qs)}")
                continue

            scores = []
            sub_result = []

            for i, sub in enumerate(sub_qs):
                score = gpt_score_subquestion(q, sub)
                if score is not None:
                    print(f"  Q{idx+1}-Sub{i+1}: 得分 = {score:.2f}")
                    scores.append(score)
                else:
                    print(f"  Q{idx+1}-Sub{i+1}: GPT 无效返回")
                    scores.append(0.0)  # or None if you prefer
                sub_result.append({"text": sub, "score": round(scores[-1], 2)})
                time.sleep(1.2)  # 控制速率，视代理而定

            avg_score = sum(scores) / len(scores)
            print(f"[Avg] Q{idx+1}: 平均得分 = {avg_score:.4f}")
            avg_score_file.write(f"Q{idx+1}: {avg_score:.4f}\n")

            result = {
                "question": q,
                "subquestions": sub_result
            }
            all_results.append(result)

        except Exception as e:
            print(f"[Error] Q{idx+1}: {q}\n{repr(e)}")
            traceback.print_exc()
            continue

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    avg_score_file.close()
    print(f"\n✅ 所有打分完成，已保存 {len(all_results)} 条记录到 {output_path}")
    print(f"✅ 平均分写入 avg_similarity_scores.txt")

if __name__ == "__main__":
    main()
