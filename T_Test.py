from scipy.stats import ttest_ind
import pandas as pd
import numpy as np

def T_Test(sample1, sample2):
    # 假设 sample1 和 sample2 是两支球队在关键时刻的得分差异数据
    t_statistic, p_value = ttest_ind(sample1, sample2)

    # 输出 t 统计量和 p-value
    print(f'T统计量: {t_statistic}, p-value: {p_value}')

    # 判断显著性水平（通常选择0.05）来决定是否拒绝原假设
    alpha = 0.05
    if p_value < alpha:
        print("拒绝原假设，差异显著")
    else:
        print("未拒绝原假设，差异不显著")


def t_test_by_momentums(momentums, data_frame, m, n):
    print(data_frame["p"])
    momentums = np.array(momentums)  # 将 Momentums 转换为 NumPy 数组

    # 进行 t 检验
    t_results = {}
    for column in data_frame.columns:
        group1_data = data_frame.loc[momentums > m, column]
        group2_data = data_frame.loc[momentums < n, column]
        # 检查数据类型是否适用于 t 检验
        if group1_data.dtype.kind not in 'biufc' or group2_data.dtype.kind not in 'biufc':
            continue

        t_statistic, p_value = ttest_ind(group1_data, group2_data)
        t_results[column] = {'t_statistic': t_statistic, 'p_value': p_value}

    return t_results




