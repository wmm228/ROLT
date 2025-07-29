import os
import pandas as pd

# 指定数据集所在的目录
data_dir = "hop_pre"  # 请替换为你的实际目录路径

# 获取所有文件
all_files = os.listdir(data_dir)

# 筛选 1-hop 训练集和测试集
one_hop_train_files = [f for f in all_files if "1hop" in f and "train" in f and f.endswith(".csv")]
one_hop_test_files = [f for f in all_files if "1hop" in f and "test" in f and f.endswith(".csv")]

# 修正 1-hop 数据的 hop 值
def fix_one_hop(file_path):
    df = pd.read_csv(os.path.join(data_dir, file_path))

    # 统一列名
    if "问题" in df.columns and "跳数" in df.columns:
        df.rename(columns={"问题": "question", "跳数": "hop"}, inplace=True)
    elif "question" in df.columns and "hop" in df.columns:
        pass  # 已经是标准列名，无需修改
    else:
        raise ValueError(f"文件 {file_path} 的列名不符合预期，请检查！")

    # 修改 hop 值为 1
    df["hop"] = 1

    # 保存回原文件
    df.to_csv(os.path.join(data_dir, file_path), index=False, encoding="utf-8")
    print(f"已修正 {file_path} 的 hop 值为 1")

# 修正 1-hop 训练集和测试集
for file in one_hop_train_files + one_hop_test_files:
    fix_one_hop(file)

# 重新筛选训练集和测试集
train_files = [f for f in all_files if "train" in f and f.endswith(".csv")]
test_files = [f for f in all_files if "test" in f and f.endswith(".csv")]

# 加载 CSV 文件并标准化列名
def load_and_standardize_csv(file_path):
    df = pd.read_csv(os.path.join(data_dir, file_path))

    # 统一列名
    if "问题" in df.columns and "跳数" in df.columns:
        df.rename(columns={"问题": "question", "跳数": "hop"}, inplace=True)
    elif "question" in df.columns and "hop" in df.columns:
        pass  # 已经是标准列名，无需修改
    else:
        raise ValueError(f"文件 {file_path} 的列名不符合预期，请检查！")

    return df

# 处理训练集
train_dataframes = [load_and_standardize_csv(file) for file in train_files]
merged_train = pd.concat(train_dataframes, ignore_index=True)
merged_train.to_csv(os.path.join(data_dir, "merged_train.csv"), index=False, encoding="utf-8")

# 处理测试集
test_dataframes = [load_and_standardize_csv(file) for file in test_files]
merged_test = pd.concat(test_dataframes, ignore_index=True)
merged_test.to_csv(os.path.join(data_dir, "merged_test.csv"), index=False, encoding="utf-8")

print(f"合并完成，训练集 {len(merged_train)} 条，测试集 {len(merged_test)} 条")
