import os
import pandas as pd

# 用于存储每个列标签的显著差异次数
column_significant_counts = {}

# 遍历数据集文件夹
for filename in os.listdir('./T_Test'):
    print(filename)
    if filename.endswith('.csv'):
        # 读取每个 CSV 文件
        file_path = os.path.join('./T_Test', filename)
        result_df = pd.read_csv(file_path)

        # 统计每个列标签的显著差异情况
        for column, count in result_df.groupby('Column')['significant difference'].sum().items():
            column_significant_counts.setdefault(column, 0)
            column_significant_counts[column] += count

# 将 column_significant_counts 转换为 DataFrame
column_significant_df = pd.DataFrame(list(column_significant_counts.items()), columns=['Column', 'Significant Counts'])

# 保存结果到 CSV 文件
column_significant_df.to_csv('column_significant_counts.csv', index=False)

# 打印结果
print(column_significant_df)