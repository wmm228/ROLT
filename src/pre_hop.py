import pandas as pd
from simpletransformers.classification import ClassificationModel

# 设置最佳模型的路径
BEST_MODEL_PATH = "/data/wxl/Decomposition_Tree/bert_base_uncased_output/best_model"

# 加载最佳模型
model = ClassificationModel(
    "bert",  
    BEST_MODEL_PATH, 
    num_labels=5,  # 确保与训练时一致
    use_cuda=True
)

# 读取 queries.tsv 文件，假设文件只有一列，存储了问题文本
queries_df = pd.read_csv('queries.tsv', sep='\t', names=['text'])

# 进行预测
predictions, raw_outputs = model.predict(queries_df['text'].tolist())

# 将预测的 hop 数添加到数据框
queries_df['predicted_hops'] = predictions

# 保存预测结果到新文件
queries_df.to_csv('queries_with_hops.tsv', sep='\t', index=False)

print("预测完成，结果已保存至 queries_with_hops.tsv")
