# Multi-hop Question Answering with MCTS and IRCoT

这是一个基于蒙特卡洛树搜索（MCTS）和链式思维检索（IRCoT）的多跳问答研究项目，专注于知识密集型多步骤问题的自动推理和回答。

## 🚀 项目特色

- **MCTS搜索算法**：使用蒙特卡洛树搜索进行智能问题分解和推理路径探索
- **IRCoT框架**：集成链式思维检索，交替进行检索和推理
- **跳数预测**：自动预测问题的复杂度（2跳/3跳），动态调整搜索策略
- **多模型支持**：支持GPT系列、FLAN-T5系列等多种语言模型
- **BGE嵌入检索**：使用BGE-M3模型进行高质量文档嵌入和检索
- **投票机制**：通过多路径推理结果投票确定最终答案

## 📁 项目结构

```
rolt/
├── main.py                 # 主执行入口，MCTS搜索逻辑
├── api.py                  # API接口，问答处理逻辑
├── llm_client.py          # LLM客户端，支持多种模型调用
├── mcts_all.py            # MCTS算法完整实现
├── bge.py                 # BGE嵌入模型封装
├── ircot-main/            # IRCoT框架完整实现
│   ├── commaqa/           # 核心推理框架
│   ├── retriever_server/  # 检索服务器
│   └── llm_server/        # LLM服务器
├── hop_pre/               # 跳数预测模型
│   ├── train_model.py     # 模型训练脚本
│   └── process_data.py    # 数据预处理
└── intro/                 # 模型评估和比较
    ├── gpt4o.py          # GPT-4o评估
    └── model_score.py    # 模型评分比较
```

## 🛠️ 安装配置

### 环境要求

- Python 3.8+
- CUDA 11.0+ (推荐)
- 8GB+ GPU内存

### 安装依赖

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 环境变量配置

```bash
# 设置GPU设备
export CUDA_VISIBLE_DEVICES="0,1,2,3"

# 设置OpenAI API (如果使用GPT模型)
export OPENAI_API_KEY="your_api_key_here"
```

## 📊 数据准备

### 支持的数据集

- **HotpotQA**: 多跳推理问答数据集
- **2WikiMultihopQA**: 基于维基百科的多跳问答
- **MuSiQue**: 多步推理问答
- **IIRC**: 不完整信息阅读理解

### 数据下载

```bash
# 下载预处理数据
cd ircot-main
./download/processed_data.sh

# 下载原始数据（用于构建检索索引）
./download/raw_data.sh
```

### 数据格式

支持标准的JSON格式，包含问题、上下文和答案：

```json
{
  "question": "Who lived longer, Muhammad Ali or Alan Turing?",
  "context": [["Muhammad Ali", ["Muhammad Ali was born in 1942..."]], ...],
  "answer": "Muhammad Ali"
}
```

## 🚀 快速开始

### 1. 基本使用

```bash
# 运行主程序（使用MCTS搜索）
python main.py

# 运行API接口
python api.py
```

### 2. 跳数预测模型训练

```bash
cd hop_pre
python train_model.py
```

### 3. 启动检索服务

```bash
# 启动Elasticsearch (需要先安装)
./bin/elasticsearch

# 启动检索服务器
cd ircot-main
uvicorn serve:app --port 8000 --app-dir retriever_server

# 构建索引
python retriever_server/build_index.py hotpotqa
```

### 4. 启动LLM服务

```bash
# 启动FLAN-T5模型服务
cd ircot-main
MODEL_NAME=flan-t5-xl uvicorn serve:app --port 8010 --app-dir llm_server
```

## ⚙️ 配置说明

### 主要参数

在 `main.py` 中可以配置的关键参数：

```python
# LLM配置
base_url = "https://api.agicto.cn/v1"
api_key = "your_api_key"

# MCTS参数
B = 4                    # 每步生成子问题数
max_hop = 3             # 最大跳数
max_iter = 15           # 最大迭代次数

# GPU配置
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
```

### 模型配置

支持的模型类型：
- **GPT系列**: gpt-3.5-turbo, gpt-4, gpt-4o
- **FLAN-T5系列**: flan-t5-base, flan-t5-large, flan-t5-xl, flan-t5-xxl
- **BERT系列**: bert-base-uncased (用于跳数预测)

## 📈 实验运行

### 运行完整实验

```bash
cd ircot-main
./reproduce.sh ircot_qa gpt-4 hotpotqa
```

### 批量评估

```bash
# 评估所有模型在HotpotQA上的表现
python runner.py ircot_qa gpt-4 hotpotqa predict --prompt_set 1
python runner.py ircot_qa gpt-4 hotpotqa summarize --prompt_set aggregate --best --eval_test --official
```

## 📊 评估指标

系统支持多种评估指标：

- **Exact Match (EM)**: 精确匹配
- **F1 Score**: F1分数
- **EMCover**: 答案包含度
- **Recall**: 召回率

### 结果示例

```
[FINAL EVALUATION]
总问题数: 500
Exact Match (EM): 0.6247
F1 Score: 0.7156
Recall: 0.7892
EMCover: 0.7234
```

## 🔧 API接口

### RESTful API

启动API服务后，可以通过HTTP请求进行问答：

```python
import requests

response = requests.post("http://localhost:8000/qa", json={
    "question": "Who lived longer, Muhammad Ali or Alan Turing?",
    "context": [...]
})

print(response.json()["answer"])
```

### Python接口

```python
from llm_client import LLMClient
from mcts_all import mcts_search

# 初始化客户端
llm_client = LLMClient(api_key="your_key", base_url="your_url")

# 执行MCTS搜索
answer = mcts_search(
    root=root_node,
    max_hop=3,
    max_iter=15,
    llm_client=llm_client,
    question=question,
    context=context
)
```

## 🎯 性能优化

### GPU加速

```python
# 多GPU并行
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

### 缓存优化

```python
# BGE嵌入缓存
bgeembedder = BGEEmbedder(
    model_name="bge_m3",
    embeddings_dir="embeddings_parts"
)
```

## 🤝 贡献指南

1. Fork 项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

## 📄 许可证

本项目基于MIT许可证开源 - 查看 [LICENSE](LICENSE) 文件了解详情

## 🙏 致谢

- [IRCoT](https://github.com/StonyBrookNLP/ircot) - 链式思维检索框架
- [CommaQA](https://github.com/allenai/CommaQA) - 复杂问答系统框架
- [BGE](https://github.com/FlagOpen/FlagEmbedding) - 高质量嵌入模型
- [HotpotQA](https://hotpotqa.github.io/) - 多跳问答数据集

## 📚 相关论文

```bibtex
@inproceedings{trivedi-etal-2023-interleaving,
    title = "Interleaving Retrieval with Chain-of-Thought Reasoning for Knowledge-Intensive Multi-Step Questions",
    author = "Trivedi, Harsh and Balasubramanian, Niranjan and Khot, Tushar and Sabharwal, Ashish",
    booktitle = "Proceedings of ACL 2023",
    year = "2023"
}
```

## 📞 联系方式

如有问题或建议，请通过以下方式联系：

- 开启 [Issue](../../issues)
- 邮件联系：[your-email@example.com]
- 项目主页：[项目链接] 