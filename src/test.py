import os
import csv
import math
import torch
import numpy as np
from sentence_transformers import SentenceTransformer

# ====== 配置 ======
tsv_file_path = "enwiki-20171001-pages-meta-current-withlinks-abstracts_clean.tsv"
model_name = "bge_m3"
output_dir = "embeddings_parts"  # 存放分批嵌入结果的目录
batch_size = 512
device = 'cuda:0'  # 指定单卡 GPU
limit_total = None  # 总共最多处理多少条数据 (None=全部)
save_every = 100000  # 每处理多少条保存一次，避免一次性塞满显存

# ====== 数据加载函数 ======
def load_paragraphs(file_path, limit=None):
    paragraphs, titles = [], []
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        for idx, row in enumerate(reader):
            if len(row) < 2:
                continue
            content = row[1]
            if 'Title:' in content and 'Text:' in content:
                try:
                    title = content.split('Title: ')[1].split(' Text: ')[0].strip()
                    text = content.split('Text: ')[1].strip()
                    paragraphs.append(text)
                    titles.append(title)
                except IndexError:
                    continue
            if limit and idx >= limit - 1:
                break
    return paragraphs, titles


# ====== 嵌入与保存 ======
def embed_and_save(texts, titles):
    print(f"[GPU] 开始处理 {len(texts)} 条数据.")
    model = SentenceTransformer(model_name, device=device)
    
    for batch_start in range(0, len(texts), save_every):
        batch_texts = texts[batch_start:batch_start + save_every]
        batch_titles = titles[batch_start:batch_start + save_every]
        print(f"[GPU] 正在处理分批: {batch_start} - {batch_start + len(batch_texts)}")

        embeddings = model.encode(batch_texts, batch_size=batch_size, show_progress_bar=True, normalize_embeddings=True)

        # 保存分批嵌入
        part_idx = batch_start // save_every
        np.save(os.path.join(output_dir, f"embeddings_part{part_idx}.npy"), embeddings)

        # 保存标题
        with open(os.path.join(output_dir, f"titles_part{part_idx}.txt"), 'w', encoding='utf-8') as f:
            for title in batch_titles:
                f.write(title + "\n")

        # 保存正文
        with open(os.path.join(output_dir, f"texts_part{part_idx}.txt"), 'w', encoding='utf-8') as f:
            for text in batch_texts:
                f.write(text + "\n")

        print(f"[GPU] 已保存 embeddings_part{part_idx}.npy 和 texts_part{part_idx}.txt")

    print(f"[GPU] 所有嵌入处理完成，共 {len(texts)} 条.")


# ====== 主流程 ======
if __name__ == "__main__":
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: 加载数据
    print("[INFO] 开始加载数据...")
    paragraphs, titles = load_paragraphs(tsv_file_path, limit=limit_total)
    total_data = len(paragraphs)
    print(f"[INFO] 加载完成，共 {total_data} 条.")

    # Step 2: 开始嵌入
    embed_and_save(paragraphs, titles)

    print("[INFO] 全部嵌入任务完成！")
