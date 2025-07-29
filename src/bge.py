import numpy as np
import os
import torch
from sentence_transformers import SentenceTransformer, util

class BGEEmbedder:
    def __init__(self, model_name="bge_m3", embeddings_dir="embeddings_parts", device=None):
        """
        初始化 BGE 模型，并加载所有嵌入文件信息
        """
        self.model_name = model_name
        self.embeddings_dir = embeddings_dir
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 加载模型
        print("[INFO] 加载 BGE-M3 模型...")
        self.model = SentenceTransformer(self.model_name, device=self.device)

        # 加载所有嵌入文件路径
        self.embedding_files, self.title_files, self.text_files = self.load_all_files()

        # 缓存嵌入，避免重复加载
        self.embeddings_cache = self.load_all_embeddings()

    def load_all_files(self):
        """
        获取嵌入、标题、文本文件的路径
        """
        embedding_files = sorted([f for f in os.listdir(self.embeddings_dir) if f.endswith(".npy")])
        title_files = sorted([f for f in os.listdir(self.embeddings_dir) if f.startswith("titles") and f.endswith(".txt")])
        text_files = sorted([f for f in os.listdir(self.embeddings_dir) if f.startswith("texts") and f.endswith(".txt")])
        print(f"[INFO] 发现 {len(embedding_files)} 个嵌入文件，{len(title_files)} 个标题文件，{len(text_files)} 个正文文件.")
        return embedding_files, title_files, text_files

    def load_all_embeddings(self):
        """
        加载所有嵌入到内存，提高查询效率
        """
        embeddings_cache = []
        for emb_file, title_file, text_file in zip(self.embedding_files, self.title_files, self.text_files):
            print(f"[INFO] 预加载嵌入文件: {emb_file}")

            embeddings = np.load(os.path.join(self.embeddings_dir, emb_file), mmap_mode='r')
            
            # 加载标题
            with open(os.path.join(self.embeddings_dir, title_file), 'r', encoding='utf-8') as f:
                titles = [line.strip() for line in f]
            
            # 加载正文
            with open(os.path.join(self.embeddings_dir, text_file), 'r', encoding='utf-8') as f:
                texts = [line.strip() for line in f]

            assert len(titles) == len(texts) == len(embeddings), "标题、正文、嵌入数量不一致！"
            
            # 缓存数据
            embeddings_cache.append((embeddings, titles, texts))
        
        return embeddings_cache

    def retrieve(self, query, top_k=5):
        """
        根据查询 `query` 进行检索，返回 `top_k` 相关文本
        """
        print(f"[INFO] 正在生成查询嵌入: {query}")
        query_embedding = self.model.encode(query, normalize_embeddings=True)
        
        all_scores, all_titles, all_texts = [], [], []

        # 遍历所有嵌入
        for embeddings, titles, texts in self.embeddings_cache:
            scores = util.cos_sim(torch.tensor(query_embedding).unsqueeze(0), torch.tensor(embeddings))[0]
            all_scores.extend(scores.tolist())
            all_titles.extend(titles)
            all_texts.extend(texts)

        # 选取 top-k 结果
        top_k_indices = np.argsort(all_scores)[-top_k:][::-1]
        top_results = [(all_scores[idx], all_titles[idx], all_texts[idx]) for idx in top_k_indices]

        return top_results
