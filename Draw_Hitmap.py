import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
# 读取数据
__path__ = './MCM/Data/Splite_Data/Splite_Data/'
df = pd.read_csv(__path__ + '2023-wimbledon-1301.csv')
print(df.columns)
df = df.drop(['match_id', 'player1', 'player2', 'elapsed_time', 'set_no', 'game_no','point_no','new_game','game_id'], axis=1)

# 计算势头与多种比赛结果之间的相关系数矩阵
correlation_matrix = df.corr()
# 你想要保留的标签
selected_labels = ['p1_sets', 'p2_sets', 'p1_games', 'p2_games','point_victor', 'p1_points_won',
       'p2_points_won','p1_break_pt_won', 'p2_break_pt_won','p1_momentum', 'p2_momentum', 'cal_m_1', 'cal_m_2',"Momentums",]
# 获取所有的x轴标签

# 只选择你想要显示的数据
selected_data = correlation_matrix[selected_labels].loc[selected_labels]

# 创建热力图
fig, ax = plt.subplots(figsize=(10, 7))
sns.heatmap(selected_data, annot=True, cmap='coolwarm', fmt=".2f", annot_kws={"size": 10}, ax=ax)

labels = [item.get_text() for item in ax.get_xticklabels()]

# 只保留"Momentums"标签
labels = [label if label in selected_labels else '' for label in labels]

# 设置新的x轴标签
ax.set_xticklabels(labels,fontsize=10)

# 设置新的y轴标签
ax.set_yticklabels(labels,fontsize=10)
# 调整标签的旋转角度
plt.xticks(rotation=45)
plt.yticks(rotation=45)

plt.title('Correlation Heatmap', fontsize=15)
plt.tight_layout()  # 调整布局以防止标签被剪切
plt.show()