import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import numpy as np
import matplotlib.patches as mpatches
import os

def Get_lasso_result(file_name, draw = True):
       df = pd.read_csv(file_name)
       df = df.drop(['match_id', 'player1', 'player2', 'elapsed_time', 'set_no', 'game_no','point_no','new_game','game_id'], axis=1)
       # 你想要保留的标签
       df['game_victor'] = df['game_victor'].replace(0,np.nan)
       df['game_victor'] = df['game_victor'].ffill()
       df['game_victor'] = df['game_victor'].fillna(0)
       df['set_victor'] = df['set_victor'].replace(0,np.nan)
       df['set_victor'] = df['set_victor'].ffill()
       df['set_victor'] = df['set_victor'].fillna(0)
       selected_labels = ['p1_sets', 'p2_sets', 'p1_games', 'p2_games', 'p1_score',
              'p2_score', 'server', 'serve_no', 'point_victor', 'p1_points_won',
              'p2_points_won', 'game_victor', 'set_victor', 'p1_ace', 'p2_ace',
              'p1_winner', 'p2_winner', 'winner_shot_type', 'p1_double_fault',
              'p2_double_fault', 'p1_unf_err', 'p2_unf_err', 'p1_net_pt', 'p2_net_pt',
              'p1_net_pt_won', 'p2_net_pt_won', 'p1_break_pt', 'p2_break_pt',
              'p1_break_pt_won', 'p2_break_pt_won', 'p1_break_pt_missed',
              'p2_break_pt_missed', 'p1_distance_run', 'p2_distance_run',
              'rally_count', 'serve_width', 'serve_depth', 'return_depth','p1_momentum', 'p2_momentum', 'cal_m_1', 'cal_m_2',]

       data = df['Momentums'].values

       change_indices = np.where(np.diff(np.sign(data)))[0]
       # 提取这些索引对应的行特征向量
       selected_features = df.loc[change_indices, selected_labels]
       # 提取对应的momentum值
       target = df.loc[change_indices, 'Momentums']
       # 数据标准化
       scaler = StandardScaler()
       features_scaled = scaler.fit_transform(selected_features)

       # 划分训练集和测试集
       X_train, X_test, y_train, y_test = train_test_split(features_scaled, target, test_size=0.2, random_state=42)

       # 创建Lasso回归模型
       lasso = Lasso(alpha=0.1)

       # 训练模型
       lasso.fit(X_train, y_train)

       if draw:
              # 假设阈值是0.1
              threshold = 0.01

              # 找出大于阈值的系数对应的label值
              selected_labels = selected_features.columns[abs(lasso.coef_) > threshold]
              # 输出系数
              print(lasso.coef_)
              print(selected_labels)
              selected_indices = [selected_features.columns.get_loc(feature) for feature in selected_labels]
              print(selected_indices)
              selected_coefs = lasso.coef_[selected_indices]
              # 将标签和系数打包成元组，然后按照系数值进行排序
              sorted_pairs = sorted(zip(selected_labels, selected_coefs), key=lambda pair: abs(pair[1]), reverse=False)
              # 将排序后的标签和系数解包为两个列表
              sorted_labels, sorted_coefs = zip(*sorted_pairs)

              # 创建两个颜色映射
              cmap_pos = mcolors.LinearSegmentedColormap.from_list("", ["lightblue","blue"])
              cmap_neg = mcolors.LinearSegmentedColormap.from_list("", ["lightpink","red"])

              # 对正数和负数分别进行映射，添加一个小的偏移量
              norm_pos = mcolors.LogNorm(vmin=0.01, vmax=max(np.abs(sorted_coefs)))
              norm_neg = mcolors.LogNorm(vmin=0.01, vmax=-min(sorted_coefs))

              # 使用颜色映射设置条形的颜色
              colors = [cmap_pos(norm_pos(np.abs(value))) if value >= 0 else cmap_neg(norm_neg(np.abs(value))) for value in sorted_coefs]
              plt.barh(sorted_labels, np.abs(sorted_coefs), color=colors)

              # 创建图例
              red_patch = mpatches.Patch(color='red', label='Positive Coefficients')
              blue_patch = mpatches.Patch(color='blue', label='Negative Coefficients')
              plt.legend(handles=[red_patch, blue_patch])


              plt.xlabel('Absolute Coefficient Value')
              plt.title('Coefficient of Lasso Regression')
              plt.grid(True)
              plt.tight_layout()
              plt.show()
       else:
              return selected_features.columns, lasso.coef_

if __name__ == "__main__":
       # 定义文件路径
       data_path = './Splite_Data/'

       # 获取文件夹下所有CSV文件
       csv_files = [f for f in os.listdir(data_path) if f.endswith('.csv')]
       csv_files = ['2023-wimbledon-1301.csv', '2023-wimbledon-1302.csv', '2023-wimbledon-1303.csv', '2023-wimbledon-1304.csv', '2023-wimbledon-1305.csv', 
                    '2023-wimbledon-1306.csv', '2023-wimbledon-1307.csv', '2023-wimbledon-1308.csv', '2023-wimbledon-1309.csv', '2023-wimbledon-1310.csv', 
                    '2023-wimbledon-1311.csv', '2023-wimbledon-1312.csv', '2023-wimbledon-1313.csv', '2023-wimbledon-1314.csv', '2023-wimbledon-1315.csv', 
                    '2023-wimbledon-1316.csv', '2023-wimbledon-1401.csv', '2023-wimbledon-1402.csv', '2023-wimbledon-1403.csv', '2023-wimbledon-1404.csv', 
                    '2023-wimbledon-1405.csv', '2023-wimbledon-1406.csv', '2023-wimbledon-1407.csv', '2023-wimbledon-1408.csv', '2023-wimbledon-1501.csv', 
                    '2023-wimbledon-1502.csv', '2023-wimbledon-1503.csv', '2023-wimbledon-1504.csv', '2023-wimbledon-1601.csv', '2023-wimbledon-1602.csv', 
                    '2023-wimbledon-1701.csv']

       #csv_files = ['2023-wimbledon-1701.csv']
       print(csv_files)

       # 创建一个空的DataFrame来保存所有结果
       result_df = pd.DataFrame(columns=['Match_Name', 'Labels', 'Coefficients'])

       i = 0
       # 循环处理每个CSV文件
       for csv_file in csv_files:
              if i >= 5:
                     break
              # 读取CSV文件
              labels, coefs = Get_lasso_result(data_path + csv_file, draw=False)
              labels = labels.to_list()

              filter_lst = ['p1_ace', 'winner_shot_type', 'p1_break_pt', 'rally_count', 'p1_unf_err', 'serve_no']
              '''
              break_t = 0
              for item in filter_lst:
                     if np.abs(coefs[labels.index(item)]) < 0.1:
                            break_t += 1
                            continue

              if break_t >= 2:
                     continue

              # 将结果添加到result_df中
              i += 1
              '''
              result_df = pd.concat([result_df, pd.DataFrame({'Match_Name': [csv_file.split('.')[0]] * len(labels),
                                                              'Labels': labels,
                                                              'Coefficients': coefs})], ignore_index=True)

       # 将结果保存到Excel文件
       result_df.to_excel('lasso_results.xlsx', index=False)


       # 读取你的数据
       df = pd.read_excel('lasso_results.xlsx')  # 请替换为你的文件路径

       '''
       # 找到需要删除的行名称
       matches_to_remove = df[df.iloc[:, 2].abs().eq(0.0)]['Labels'].unique()

       # 删除符合条件的比赛数据
       df = df[~df['Labels'].isin(matches_to_remove)]

       # 重新排列索引
       df = df.reset_index(drop=True)

       # 保存修改后的数据
       df.to_excel('modified_data.xlsx', index=False)
       '''

       filtered_df = df[df['Labels'].isin(filter_lst)]

       # 将结果保存到新的 Excel 文件
       filtered_df.to_excel('filtered_results.xlsx', index=False)