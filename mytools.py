import csv
import matplotlib
matplotlib.use('TkAgg')  # 选择一个合适的后端，比如 TkAgg
import matplotlib.pyplot as plt


def Draw_Line_Graph(y_axis, x_name, y_name, title):
    x_axis = range(1, len(y_axis) + 1)

    # 绘制曲线图
    plt.plot(x_axis, y_axis)

    # 添加标签和标题
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.title(title)

    # 显示网格
    plt.grid(True)

    # 显示图形
    plt.show()

