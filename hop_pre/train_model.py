import pandas as pd
import random
from simpletransformers.classification import ClassificationModel, ClassificationArgs
import logging
import numpy as np
import os
import time
import psutil
import torch.nn.functional as F
import json

os.environ["CUDA_VISIBLE_DEVICES"] = "1"   # 根据自己的 GPU 情况进行修改
TYPE = 'bert'
MODEL = "/data/wxl/Decomposition_Tree/bert_base_uncased"

os.makedirs('./hop', exist_ok=True)

# ========== 1. 数据集加载 ==========
import pandas as pd

# 读取数据集
train_df = pd.read_csv('./new_dataset/new_train.csv')
test_df = pd.read_csv('./new_dataset/new_test.csv')

# 打乱训练数据
train_df = train_df.sample(frac=1).reset_index(drop=True)

# 划分训练集和验证集（90% 训练, 10% 验证）
split = int(len(train_df) * 0.9)
valid_df = train_df.iloc[split:].reset_index(drop=True)
train_df = train_df.iloc[:split].reset_index(drop=True)

# **重新映射 labels: 2 → 0, 3 → 1**
train_df['labels'] = train_df['hop'].replace({2: 0, 3: 1}).astype(int)
valid_df['labels'] = valid_df['hop'].replace({2: 0, 3: 1}).astype(int)
test_df['labels'] = test_df['hop'].replace({2: 0, 3: 1}).astype(int)

# **确保 labels 只包含整数**
assert train_df['labels'].isin([0, 1]).all(), "训练集 labels 仍然存在错误！"
assert valid_df['labels'].isin([0, 1]).all(), "验证集 labels 仍然存在错误！"
assert test_df['labels'].isin([0, 1]).all(), "测试集 labels 仍然存在错误！"

# 丢弃原 hop 列, 并重命名 question -> text
train_df = train_df[['question', 'labels']].rename(columns={'question': 'text'})
valid_df = valid_df[['question', 'labels']].rename(columns={'question': 'text'})
test_df = test_df[['question', 'labels']].rename(columns={'question': 'text'})

# 打印 labels 分布，确保转换正确
print("训练集 labels 分布:\n", train_df['labels'].value_counts())
print("验证集 labels 分布:\n", valid_df['labels'].value_counts())
print("测试集 labels 分布:\n", test_df['labels'].value_counts())

print("Training Data:\n", train_df.head())
print("Validation Data:\n", valid_df.head())
print("Testing Data:\n", test_df.head())

# ========== 2. 模型与训练配置 ==========
model_args = ClassificationArgs()
model_args.train_batch_size = 64    # 训练批次
model_args.eval_batch_size = 32     # 评估批次
model_args.n_gpu = 1
model_args.save_best_model = True
model_args.save_eval_checkpoints = False
model_args.save_model_every_epoch = False
model_args.save_optimizer_and_scheduler = False
model_args.save_steps = -1
model_args.evaluate_during_training_verbose = True
model_args.overwrite_output_dir = True
model_args.num_train_epochs = 10
model_args.output_dir = MODEL + '_output/'
model_args.evaluate_during_training = True
model_args.use_early_stopping = True
model_args.early_stopping_consider_epochs = True
model_args.early_stopping_patience = 5
model_args.best_model_dir = MODEL + '_output/best_model/'
model_args.evaluate_during_training_steps = -1
model_args.use_multiprocessing = False
model_args.dataloader_num_workers = 0
model_args.process_count = 1
model_args.use_multiprocessing_for_evaluation = False

# **确保 num_labels=2**
NUM_LABELS = 2  # 现在 hop 只有 2 个类别

# 训练模型
model = ClassificationModel(
    TYPE,
    MODEL, 
    num_labels=NUM_LABELS,  # **确保 num_labels=2**
    use_cuda=True,
    args=model_args
)

# ========== 3. 训练模型 ==========
print('*' * 30, 'Start Training', '*' * 30)
time1 = time.time()
model.train_model(train_df, eval_df=valid_df)
time2 = time.time()
train_time = time2 - time1

# ========== 4. 加载最佳模型并验证 ==========
model = ClassificationModel(
    TYPE, 
    MODEL + '_output/best_model', 
    num_labels=NUM_LABELS,  # **确保 num_labels=2**
    use_cuda=True,
    args=model_args
)

print('*' * 30, 'Start Validation', '*' * 30)
time3 = time.time()
result, model_outputs, wrong_predictions = model.eval_model(valid_df, verbose=True)
time4 = time.time()
valid_time = time4 - time3

# ========== 4.1 计算 Top-1 准确率 ==========
correct = 0
with open('hop/valid.txt', 'w', encoding='utf-8') as f:
    for i, logits in enumerate(model_outputs):
        truth = valid_df['labels'][i]
        order = np.flipud(np.argsort(logits))  # 从高到低排序索引
        if truth == order[0]:
            correct += 1
        f.write(str(order[0]) + '\n')

P_valid = correct / len(valid_df)
print("Validation Precision:", P_valid)

# ========== 5. 测试集评估 ==========
print('*' * 30, 'Start Testing', '*' * 30)
time5 = time.time()
result, model_outputs, wrong_predictions = model.eval_model(test_df, verbose=True)
time6 = time.time()
test_time = time6 - time5

correct = 0
for i, logits in enumerate(model_outputs):
    truth = test_df['labels'][i]
    order = np.flipud(np.argsort(logits))
    if truth == order[0]:
        correct += 1
P_test = correct / len(test_df)
print("Test Precision:", P_test)

# ========== 5.1 保存测试集预测结果 ==========
text_list = list(test_df['text'])
tmp_test_df = pd.DataFrame({"text": text_list, "labels": [0]*len(text_list)})

result, model_outputs, wrong_predictions = model.eval_model(tmp_test_df, verbose=True)
with open('hop/test.txt', 'w', encoding='utf-8') as f:
    for logits in model_outputs:
        order = np.flipud(np.argsort(logits))
        f.write(str(order[0]) + '\n')

print('Train Time:', train_time)
print('Valid Time:', valid_time)
print('Test Time:', test_time)
