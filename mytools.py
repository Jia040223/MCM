import csv
import matplotlib
matplotlib.use('TkAgg')  # 选择一个合适的后端，比如 TkAgg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

def Draw_Line_Graph(y_axis, x_name, y_name, title):
    x_axis = range(1, len(y_axis) + 1)

    # 绘制曲线图
    plt.xticks(np.arange(min(x_axis) - 1, max(x_axis) + 10, 20))
    plt.plot(x_axis, y_axis, linewidth=1.0)

    # 添加标签和标题
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.title(title)

    # 显示网格
    plt.grid(True)

    # 显示图形
    plt.show()

def Draw_Filled_Line_Graph(y_axis, x_name, y_name, title):
    x_axis = range(1, len(y_axis) + 1)

    # 创建一个与 y 相同长度的零数组
    y_axis = np.array(y_axis)
    zeros = np.zeros_like(y_axis)

    # 判断小于0和大于等于0的部分
    mask_negative = y_axis < 0
    mask_positive = y_axis >= 0

    plt.xticks(np.arange(min(x_axis)-1, max(x_axis) + 10, 20))

    # 画折线图
    plt.plot(x_axis, y_axis, label='折线图', linewidth=0.05)

    # 用红色填充小于0的部分
    plt.fill_between(x_axis, y_axis, where=mask_negative, interpolate=True, color=(255/255, 0/255, 0/255), alpha=0.3, label='小于0')

    # 用紫色填充大于等于0的部分
    plt.fill_between(x_axis, y_axis, where=mask_positive, interpolate=True, color=(0/255, 0/255, 255/255), alpha=0.3, label='大于等于0')

    # 添加标签和标题
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.title(title)

    # 显示网格
    plt.grid(True)

    # 显示图形
    plt.show()

def Add_csv_column(file_path, new_column_data, new_column_name, save_path):
    # 读取CSV文件
    df = pd.read_csv(file_path)

    # 添加新列到DataFrame
    df[new_column_name] = new_column_data

    # 将带有新列的DataFrame保存回CSV文件
    df.to_csv(save_path, index=False)


def Radar_Chart(csv_folder_path):
    # 获取文件夹中所有CSV文件的文件名
    csv_files = [f for f in os.listdir(csv_folder_path) if f.endswith('.csv')]

    # 创建极坐标子图
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})

    color_map = plt.cm.get_cmap('Set3')

    # 遍历每个CSV文件
    for i, csv_file in enumerate(csv_files):
        # 读取CSV文件
        df = pd.read_csv(os.path.join(csv_folder_path, csv_file))

        # 选择需要绘制雷达图的列
        selected_rows = ['p1_score', 'point_victor', 'p1_points_won', 'p1_games', 'p1_ace', 'p1_winner', 'p2_unf_err']
        selected_column = 'p-value'

        # 根据列名获取对应的数据
        data = []
        for row_name in selected_rows:
            row_data = df.loc[df['Column'] == row_name, selected_column].values[0]
            data.append(row_data)
        data = [(1 - i) for i in data]
        print(data)

        # 获取雷达图的标签
        labels = selected_rows

        # 获取雷达图的角度
        angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)

        # 将数据首尾相连，使雷达图闭合
        values = np.concatenate((data, [data[0]]))
        angles = np.concatenate((angles, [angles[0]]))

        # 绘制雷达图，每个图使用不同颜色
        color = plt.cm.viridis(i / len(csv_files))
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=csv_file[:-4], alpha=0.7, color=color)

        # 填充折线下方的颜色
        ax.fill(angles, values, alpha=0.3, color=color)

    # 添加标签
    ax.set_thetagrids(angles[:-1] * 180 / np.pi, labels)

    # 显示图例
    ax.legend(bbox_to_anchor=(1, 0), loc='lower left')
    # 显示图形
    plt.show()

def Judge_final_win(df):
    # 获取最后一行数据
    last_row = df.iloc[-1]

    # 比较条件并输出结果
    if last_row['p1_sets'] > last_row['p2_sets']:
        result = 1
    elif last_row['p1_sets'] < last_row['p2_sets']:
        result = 0
    else:  # 如果 sets 相等，则比较 games
        if last_row['p1_games'] > last_row['p2_games']:
            result = 1
        elif last_row['p1_games'] < last_row['p2_games']:
            result = 0
        else:  # 如果 games 也相等，则比较 scores
            if last_row['p1_score'] > last_row['p2_score']:
                result = 1
            elif last_row['p1_score'] < last_row['p2_score']:
                result = 0
            else:
                result = -1  # 如果 scores 也相等，你可以定义其他的处理方式


import csv

def save_to_csv(list1, list2, file_path):
    data_to_write = list(zip(list1, list2))

    # 使用 csv 模块写入数据到 CSV 文件
    with open(file_path, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerows(data_to_write)


def plot_csv_files(folder_path):
    files = os.listdir(folder_path)

    # 读取并绘制每个 CSV 文件的第一列数据
    for file in files:
        if file.endswith('.csv'):
            # 拼接完整的文件路径
            file_path = os.path.join(folder_path, file)

            # 从文件名中提取参数信息作为图例
            legend_info = tuple(map(float, file.replace('result', '').replace('.csv', '').split('-')))[0]

            # 读取 CSV 文件
            df = pd.read_csv(file_path)

            # 绘制曲线，使用参数信息作为图例
            plt.plot(df.iloc[:, 0], label=legend_info)

    # 添加图例和标签
    plt.legend(title=r'Initial P1')
    plt.xlabel('Points')
    plt.ylabel('Momentums')
    plt.title('Momentums over Points')

    # 显示图形
    plt.show()

if __name__ == "__main__":
    plot_csv_files("./Sensitivity Analysis/")